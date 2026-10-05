"""Construction des composants à partir de la configuration (.env)."""
from __future__ import annotations

from dotenv import load_dotenv

from .agent import SupportAgent
from .llm import LLMClient
from .retrieval import BM25Retriever, DenseRetriever, HybridRetriever, Retriever, load_chunks
from .tools import Toolbox


def build_retriever(method: str, llm: LLMClient | None = None) -> Retriever:
    chunks = load_chunks()
    if method == "bm25":
        return BM25Retriever(chunks)
    llm = llm or LLMClient()
    dense = DenseRetriever(chunks, llm)
    if method == "dense":
        return dense
    if method == "hybrid":
        return HybridRetriever([BM25Retriever(chunks), dense])
    raise ValueError(f"méthode inconnue : {method}")


def build_agent(method: str = "hybrid") -> SupportAgent:
    load_dotenv()
    llm = LLMClient()
    return SupportAgent(llm, Toolbox(build_retriever(method, llm)))
