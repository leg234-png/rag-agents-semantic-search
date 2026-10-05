"""Découpage du corpus et trois méthodes de recherche : BM25, dense (embeddings) et hybride (RRF)."""
from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "data" / "docs"
CACHE_DIR = ROOT / "data" / "cache"

STOPWORDS = set("""au aux avec ce ces dans de des du elle en et eux il je la le les leur lui ma mais me même mes moi
mon ne nos notre nous on ou par pas pour qu que qui sa se ses son sur ta te tes toi ton tu un une vos votre vous
c d j l à m n s t y été est sont ai as a avons avez ont être avoir fait faire peut puis peux si est-ce quel quelle
quels quelles combien comment quand quoi ça cela cette cet plus""".split())


@dataclass(frozen=True)
class Chunk:
    id: str        # ex : "annulation#bareme-des-frais-d-annulation"
    doc: str
    title: str
    section: str
    text: str

    @property
    def full_text(self) -> str:
        return f"{self.title} - {self.section}\n{self.text}"


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", strip_accents(s.lower())).strip("-")


def load_chunks(docs_dir: Path = DOCS_DIR) -> list[Chunk]:
    """Un chunk = une section `##` d'un document Markdown (unités de sens courtes et citables)."""
    chunks = []
    for path in sorted(docs_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title = re.search(r"^# (.+)$", text, re.M).group(1).strip()
        for m in re.finditer(r"^## (.+?)\n(.*?)(?=^## |\Z)", text, re.M | re.S):
            section, body = m.group(1).strip(), m.group(2).strip()
            chunks.append(Chunk(f"{path.stem}#{slugify(section)}", path.stem, title, section, body))
    return chunks


def tokenize(text: str) -> list[str]:
    toks = re.findall(r"[a-z0-9]+", strip_accents(text.lower()))
    # racinisation légère : retire les pluriels simples
    return [t[:-1] if len(t) > 4 and t.endswith("s") else t for t in toks if t not in STOPWORDS]


class Retriever(Protocol):
    name: str

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]: ...


class BM25Retriever:
    name = "bm25"

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c.full_text) for c in chunks])

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top]


class Embedder(Protocol):
    embed_model: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class DenseRetriever:
    """Similarité cosinus entre embeddings. Les embeddings du corpus sont mis en cache sur disque."""
    name = "dense"

    def __init__(self, chunks: list[Chunk], embedder: Embedder, cache_dir: Path | None = CACHE_DIR):
        self.chunks, self.embedder = chunks, embedder
        self.matrix = self._normalize(self._corpus_embeddings(cache_dir))

    @staticmethod
    def _normalize(m: np.ndarray) -> np.ndarray:
        return m / np.linalg.norm(m, axis=1, keepdims=True)

    def _corpus_embeddings(self, cache_dir: Path | None) -> np.ndarray:
        texts = [c.full_text for c in self.chunks]
        key = hashlib.sha256(("\n\n".join(texts) + self.embedder.embed_model).encode()).hexdigest()[:16]
        path = cache_dir / f"emb_{slugify(self.embedder.embed_model)}_{key}.npy" if cache_dir else None
        if path and path.exists():
            return np.load(path)
        m = np.array(self.embedder.embed(texts), dtype=np.float32)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, m)
        return m

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        q = self._normalize(np.array(self.embedder.embed([query]), dtype=np.float32))[0]
        scores = self.matrix @ q
        top = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top]


class HybridRetriever:
    """Reciprocal Rank Fusion : score = somme de 1 / (rrf_k + rang) sur les retrievers combinés."""
    name = "hybrid_rrf"

    def __init__(self, retrievers: list[Retriever], rrf_k: int = 60, depth: int = 20):
        self.retrievers, self.rrf_k, self.depth = retrievers, rrf_k, depth

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        fused: dict[str, float] = {}
        by_id: dict[str, Chunk] = {}
        for r in self.retrievers:
            for rank, (chunk, _) in enumerate(r.search(query, self.depth)):
                fused[chunk.id] = fused.get(chunk.id, 0.0) + 1.0 / (self.rrf_k + rank + 1)
                by_id[chunk.id] = chunk
        top = sorted(fused, key=fused.get, reverse=True)[:k]
        return [(by_id[i], fused[i]) for i in top]
