"""Tests sans réseau : faux embedder (hachage de mots) et faux LLM scripté."""
from __future__ import annotations

import hashlib
from collections import deque

import numpy as np
import pytest
from fastapi.testclient import TestClient

from rag_agents import api
from rag_agents.agent import MAX_STEPS, AgentAction, SupportAgent
from rag_agents.retrieval import (BM25Retriever, DenseRetriever, HybridRetriever, load_chunks, slugify,
                                  tokenize)
from rag_agents.tools import CancellationArgs, Toolbox, cancellation_fee

CHUNKS = load_chunks()


class HashEmbedder:
    embed_model = "fake-hash"

    def embed(self, texts):
        out = []
        for t in texts:
            v = np.zeros(256)
            for tok in tokenize(t):
                v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % 256] += 1
            out.append((v + 1e-6).tolist())
        return out


class FakeLLM:
    def __init__(self, actions):
        self.q = deque(actions)
        self.prompts = []

    def chat_json(self, system, user, schema):
        self.prompts.append(user)
        return self.q.popleft()


def test_chunking():
    assert len(CHUNKS) == 41
    assert len({c.id for c in CHUNKS}) == len(CHUNKS)
    assert slugify("Barème des frais d'annulation") == "bareme-des-frais-d-annulation"


def test_bm25_finds_exact_terms():
    hits = BM25Retriever(CHUNKS).search("frais d'annulation barème", 3)
    assert "annulation#bareme-des-frais-d-annulation" in [c.id for c, _ in hits]


def test_dense_and_hybrid(tmp_path):
    dense = DenseRetriever(CHUNKS, HashEmbedder(), cache_dir=tmp_path)
    assert list(tmp_path.glob("*.npy"))  # cache écrit
    hyb = HybridRetriever([BM25Retriever(CHUNKS), dense])
    ids = [c.id for c, _ in hyb.search("mineurs non accompagnés service", 5)]
    assert "enfants#mineurs-non-accompagnes" in ids and len(ids) == len(set(ids))


@pytest.mark.parametrize("days,fare,rate", [(45, "classic", 0.10), (30, "classic", 0.30), (15, "basic", 0.30),
                                            (14, "classic", 0.50), (7, "classic", 0.50), (6, "classic", 1.0),
                                            (5, "flex", 0.0), (0, "flex", 1.0)])
def test_cancellation_scale(days, fare, rate):
    out = cancellation_fee(CancellationArgs(price_eur=200, days_before_departure=days, fare=fare))
    assert out["fee_rate"] == rate and out["refund_eur"] == pytest.approx(200 * (1 - rate))


def test_tool_errors_are_returned_not_raised():
    tb = Toolbox(BM25Retriever(CHUNKS))
    assert "error" in tb.call("compute_cancellation_fee", {"price_eur": -5, "days_before_departure": 3})
    assert "error" in tb.call("hack_the_planet", {})


def make_agent(actions):
    return SupportAgent(FakeLLM(actions), Toolbox(BM25Retriever(CHUNKS)))


def test_agent_search_then_answer_with_valid_citation():
    agent = make_agent([
        AgentAction(action="search_docs", args={"query": "mineurs non accompagnés"}),
        AgentAction(action="final_answer", answer="Oui, avec le service à 70 €.",
                    citations=["enfants#mineurs-non-accompagnes"]),
    ])
    res = agent.ask("Ma fille de 8 ans peut-elle voyager seule ?")
    assert res.verified and res.citations == ["enfants#mineurs-non-accompagnes"]
    assert "Observation" in agent.llm.prompts[1]  # l'observation est bien renvoyée au LLM


def test_agent_invented_citation_is_flagged():
    agent = make_agent([AgentAction(action="final_answer", answer="Gratuit !", citations=["bagages#inventee"])])
    res = agent.ask("Le bagage est-il gratuit ?")
    assert not res.verified and res.citations == []


def test_agent_tool_call_and_source_citation():
    agent = make_agent([
        AgentAction(action="compute_cancellation_fee", args={"price_eur": 480, "days_before_departure": 20}),
        AgentAction(action="final_answer", answer="Vous récupérez 336 €.",
                    citations=["annulation#bareme-des-frais-d-annulation"]),
    ])
    res = agent.ask("480 €, annulation à J-20 ?")
    assert res.verified and res.steps[0]["observation"]["refund_eur"] == 336.0


def test_agent_step_budget():
    agent = make_agent([AgentAction(action="search_docs", args={"query": "x"})] * (MAX_STEPS + 1))
    res = agent.ask("question sans fin")
    assert not res.verified and len(res.steps) == MAX_STEPS


def test_api_search_bm25():
    client = TestClient(api.app)
    assert client.get("/health").json() == {"status": "ok"}
    r = client.post("/search", json={"query": "chat en cabine", "k": 3, "method": "bm25"})
    assert r.status_code == 200 and r.json()[0]["id"] == "animaux#en-cabine"
