"""Évaluation de la recherche (BM25 / dense / hybride) et de l'agent.

    python -m rag_agents.evaluate retrieval --methods bm25            # sans clé API
    python -m rag_agents.evaluate retrieval --methods bm25 dense hybrid
    python -m rag_agents.evaluate agent --method hybrid

Les 30 questions de test sont formulées comme un client les poserait : elles reprennent rarement
les mots exacts de la documentation, ce qui met en évidence l'intérêt de la recherche sémantique.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from .retrieval import Retriever

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

# Cas qui exigent un outil : (question, outil attendu, champ de l'observation, valeur attendue).
# On vérifie que l'agent appelle le bon outil AVEC les bons arguments (valeur calculée correcte).
TOOL_CASES = [
    ("J'ai payé 480 € et j'annule 20 jours avant le départ, combien vais-je récupérer ?",
     "compute_cancellation_fee", "refund_eur", 336.0),
    ("Billet Classic à 250 €, j'annule 3 jours avant : combien je perds ?",
     "compute_cancellation_fee", "fee_eur", 250.0),
    ("Mon tarif Flex coûte 300 € et je veux annuler 5 jours avant, quels frais ?",
     "compute_cancellation_fee", "fee_eur", 0.0),
    ("Combien font 200 euros en livres sterling d'après vos taux ?", "convert_currency", "amount", 170.0),
]


def load_questions() -> list[dict]:
    return [json.loads(line) for line in (ROOT / "data" / "eval_questions.jsonl").read_text().splitlines()]


def retrieval_metrics(retriever: Retriever, questions: list[dict], ks=(1, 3, 5)) -> dict:
    recalls = {k: [] for k in ks}
    rr = []
    for q in questions:
        ranked = [c.id for c, _ in retriever.search(q["question"], max(ks))]
        rel = set(q["relevant"])
        for k in ks:
            recalls[k].append(len(rel & set(ranked[:k])) / len(rel))
        pos = next((i for i, cid in enumerate(ranked) if cid in rel), None)
        rr.append(0.0 if pos is None else 1.0 / (pos + 1))
    out = {f"recall@{k}": round(float(np.mean(v)), 3) for k, v in recalls.items()}
    out["mrr@5"] = round(float(np.mean(rr)), 3)
    return out


def run_retrieval(methods: list[str]) -> dict:
    from .factory import build_retriever

    load_dotenv()
    qs = load_questions()
    res = {m: retrieval_metrics(build_retriever(m), qs) for m in methods}
    print(pd.DataFrame(res).T.to_string())
    return res


def run_agent(method: str) -> dict:
    from .factory import build_agent

    agent = build_agent(method)
    rows = []
    for q in load_questions():
        r = agent.ask(q["question"])
        rows.append({"question": q["question"], "cites_relevant": bool(set(r.citations) & set(q["relevant"])),
                     "verified": r.verified, "n_steps": len(r.steps), "answer": r.answer})
        print(("OK  " if rows[-1]["cites_relevant"] else "ERR ") + q["question"])
    tool_rows = []
    for question, tool, field, expected in TOOL_CASES:
        r = agent.ask(question)
        obs = [st["observation"] for st in r.steps if st["action"] == tool]
        tool_rows.append({"question": question, "used_tool": bool(obs),
                          "value_ok": any(o.get(field) == expected for o in obs), "answer": r.answer})
        print(("OK  " if tool_rows[-1]["used_tool"] and tool_rows[-1]["value_ok"] else "ERR ") + question)
    df, tdf = pd.DataFrame(rows), pd.DataFrame(tool_rows)
    REPORTS.mkdir(exist_ok=True)
    pd.concat([df, tdf]).to_csv(REPORTS / "agent_answers.csv", index=False)
    return {"retriever": method, "model": agent.llm.model,
            "citation_hit_rate": round(df["cites_relevant"].mean(), 3),
            "verified_rate": round(df["verified"].mean(), 3),
            "avg_steps": round(df["n_steps"].mean(), 2),
            "tool_selection_rate": round(tdf["used_tool"].mean(), 3),
            "tool_answer_accuracy": round((tdf["used_tool"] & tdf["value_ok"]).mean(), 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("retrieval")
    r.add_argument("--methods", nargs="+", default=["bm25"], choices=["bm25", "dense", "hybrid"])
    a = sub.add_parser("agent")
    a.add_argument("--method", default="hybrid", choices=["bm25", "dense", "hybrid"])
    args = ap.parse_args()

    REPORTS.mkdir(exist_ok=True)
    if args.cmd == "retrieval":
        res = run_retrieval(args.methods)
        path = REPORTS / "retrieval_metrics.json"
    else:
        res = run_agent(args.method)
        path = REPORTS / "agent_metrics.json"
    path.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
