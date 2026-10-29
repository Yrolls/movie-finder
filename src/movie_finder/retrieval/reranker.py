"""Cross-encoder reranking of a retrieved shortlist (Lecture 8: retrieve broad -> rerank precise).

The first-pass retriever (BM25 + bi-encoder, fused with RRF) scores query and passage
separately. A cross-encoder reads the query and each candidate passage TOGETHER, which is
slower but sharper, so it only sees the top-N shortlist. The eval showed many correct films
sitting at rank 2-8 -- exactly what reranking can lift to rank 1.

Default model: cross-encoder/ms-marco-MiniLM-L-6-v2 (English, ~90 MB, CPU-friendly). Queries
are LLM-rewritten into English before search, so an English model fits.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from ..schemas import SearchHit

MAX_PASSAGE_WORDS = 256


class Reranker(Protocol):
    def scores(self, query: str, passages: Sequence[str]) -> list[float]: ...


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder  # heavy import, load lazily

        self.model_name = model_name
        self._model = CrossEncoder(model_name, max_length=512)

    def scores(self, query: str, passages: Sequence[str]) -> list[float]:
        pairs = [(query, " ".join(p.split()[:MAX_PASSAGE_WORDS])) for p in passages]
        return [float(s) for s in self._model.predict(pairs, show_progress_bar=False)]


def rerank(reranker: Reranker, query: str, hits: list[SearchHit], passages: Sequence[str]) -> list[SearchHit]:
    """Reorder hits by cross-encoder score. Hits keep their dense similarity in `.score`
    (the min_score floor and the UI use it); only the ORDER changes."""
    if len(hits) < 2:
        return hits
    s = reranker.scores(query, passages)
    order = sorted(range(len(hits)), key=lambda i: s[i], reverse=True)
    return [hits[i] for i in order]
