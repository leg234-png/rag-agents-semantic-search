"""Agent ReAct orchestré avec LangGraph : il choisit ses outils, observe, puis répond en citant ses sources.

    decide (LLM) ──(action = outil)──> act (exécution de l'outil) ──> decide ...
        │
        └──(final_answer ou budget d'étapes atteint)──> finish (vérification des citations) ──> FIN

Garde-fous :
- budget de MAX_STEPS actions ;
- les citations doivent correspondre à des sources réellement observées (sinon elles sont retirées
  et la réponse est signalée comme non vérifiée) ;
- arguments d'outils validés par Pydantic, les erreurs sont renvoyées à l'agent.
"""
from __future__ import annotations

import json
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from .llm import LLM
from .tools import Toolbox

MAX_STEPS = 5

SYSTEM = """Tu es l'assistant du service client d'Atlas Voyages (agence fictive).
Tu réponds en français, de façon précise et concise, UNIQUEMENT à partir des informations obtenues
avec tes outils. Si l'information est introuvable, dis-le.

Outils disponibles :
{tools}

À chaque étape, renvoie UNE action :
- {{"thought": "...", "action": "<nom_outil>", "args": {{...}}}} pour appeler un outil ;
- {{"thought": "...", "action": "final_answer", "answer": "...", "citations": ["<id>", ...]}} pour répondre.
Cherche dans la documentation avant de répondre à une question sur les règles. Utilise
compute_cancellation_fee pour tout calcul de frais d'annulation et convert_currency pour les conversions.
Les citations sont les identifiants (champ "id") des passages ou outils qui justifient la réponse."""


class AgentAction(BaseModel):
    thought: str = ""
    action: Literal["search_docs", "compute_cancellation_fee", "convert_currency", "final_answer"]
    args: dict[str, Any] = Field(default_factory=dict)
    answer: str | None = None
    citations: list[str] = Field(default_factory=list)


class AgentResult(BaseModel):
    answer: str
    citations: list[str]
    verified: bool
    steps: list[dict]


class AgentState(TypedDict, total=False):
    question: str
    steps: list[dict]
    seen_sources: list[str]
    last_action: AgentAction
    result: AgentResult


class SupportAgent:
    def __init__(self, llm: LLM, toolbox: Toolbox):
        self.llm, self.toolbox = llm, toolbox
        self.system = SYSTEM.format(tools=toolbox.describe())
        self.graph = self._build()

    def _scratchpad(self, s: AgentState) -> str:
        lines = [f"Question : {s['question']}"]
        for i, st in enumerate(s.get("steps", []), 1):
            lines.append(f"\nÉtape {i} - action {st['action']} {json.dumps(st['args'], ensure_ascii=False)}")
            lines.append("Observation : " + json.dumps(st["observation"], ensure_ascii=False))
        remaining = MAX_STEPS - len(s.get("steps", []))
        lines.append(f"\nIl te reste {remaining} action(s). Quelle est la prochaine action ?")
        return "\n".join(lines)

    def decide(self, s: AgentState) -> AgentState:
        action = self.llm.chat_json(self.system, self._scratchpad(s), AgentAction)
        return {"last_action": action}

    def act(self, s: AgentState) -> AgentState:
        a = s["last_action"]
        obs = self.toolbox.call(a.action, a.args)
        seen = list(s.get("seen_sources", []))
        seen += [r["id"] for r in obs.get("results", [])]
        if "source" in obs:
            seen.append(obs["source"])
        step = {"action": a.action, "args": a.args, "thought": a.thought, "observation": obs}
        return {"steps": s.get("steps", []) + [step], "seen_sources": seen}

    def finish(self, s: AgentState) -> AgentState:
        a = s["last_action"]
        if a.action != "final_answer":  # budget épuisé
            res = AgentResult(answer="Je n'ai pas pu aboutir à une réponse fiable, un conseiller va reprendre "
                              "votre demande.", citations=[], verified=False, steps=s.get("steps", []))
            return {"result": res}
        seen = set(s.get("seen_sources", []))
        valid = [c for c in a.citations if c in seen]
        verified = bool(valid) and len(valid) == len(a.citations)
        return {"result": AgentResult(answer=a.answer or "", citations=valid, verified=verified,
                                      steps=s.get("steps", []))}

    @staticmethod
    def route(s: AgentState) -> str:
        if s["last_action"].action == "final_answer" or len(s.get("steps", [])) >= MAX_STEPS:
            return "finish"
        return "act"

    def _build(self):
        g = StateGraph(AgentState)
        g.add_node("decide", self.decide)
        g.add_node("act", self.act)
        g.add_node("finish", self.finish)
        g.set_entry_point("decide")
        g.add_conditional_edges("decide", self.route, {"act": "act", "finish": "finish"})
        g.add_edge("act", "decide")
        g.add_edge("finish", END)
        return g.compile()

    def ask(self, question: str) -> AgentResult:
        return self.graph.invoke({"question": question, "steps": [], "seen_sources": []},
                                 {"recursion_limit": 4 * MAX_STEPS})["result"]
