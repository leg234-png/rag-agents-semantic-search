"""API REST (FastAPI).

    uvicorn rag_agents.api:app --reload
    curl -X POST localhost:8000/search -H "Content-Type: application/json" -d '{"query": "valise trop lourde"}'
    curl -X POST localhost:8000/ask -H "Content-Type: application/json" -d '{"question": "Puis-je annuler ?"}'
"""
from __future__ import annotations

import os
from functools import lru_cache

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .agent import AgentResult

app = FastAPI(title="Atlas Voyages - RAG & agents", version="1.0")
METHOD = os.getenv("RETRIEVER", "hybrid")


class SearchRequest(BaseModel):
    query: str
    k: int = Field(5, ge=1, le=20)
    method: str = "bm25"


class AskRequest(BaseModel):
    question: str


@lru_cache
def retriever(method: str):
    from .factory import build_retriever
    return build_retriever(method)


@lru_cache
def agent():
    from .factory import build_agent
    return build_agent(METHOD)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/search")
def search(req: SearchRequest) -> list[dict]:
    return [{"id": c.id, "section": f"{c.title} - {c.section}", "score": round(s, 4), "text": c.text}
            for c, s in retriever(req.method).search(req.query, req.k)]


@app.post("/ask", response_model=AgentResult)
def ask(req: AskRequest) -> AgentResult:
    return agent().ask(req.question)
