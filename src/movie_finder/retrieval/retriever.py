"""Movie-level search on top of chunk-level scores.

A movie's score is its BEST chunk's score; that chunk is returned as the citation.

modes
- "dense"  : multilingual embeddings + cosine, with a `min_score` floor   (the app's default)
- "bm25"   : keyword BM25 over the same chunks                            (the baseline)
- "hybrid" : reciprocal-rank fusion of dense + BM25 movie rankings        (an option to evaluate)
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np

from ..schemas import Movie, SearchHit
from .bm25 import BM25Index
from .embedder import Embedder
from .vector_store import VectorStore

Mode = Literal["dense", "bm25", "hybrid"]
_RRF_K = 60          # standard RRF constant
_POOL = 200          # how many top chunks to look at before collapsing to movies


class Retriever:
    def __init__(
        self,
        movies: Sequence[Movie],
        store: VectorStore,
        embedder: Embedder | None,
        *,
        min_score: float = 0.0,
        build_bm25: bool = True,
    ):
        self.movies = {m.movie_id: m for m in movies}
        self.store = store
        self.embedder = embedder
        self.min_score = min_score
        self._chunk_movie = np.array([c.movie_id for c in store.chunks])
        self._chunk_year = np.array([self.movies[c.movie_id].year for c in store.chunks])
        self.bm25 = BM25Index([c.text for c in store.chunks]) if build_bm25 else None

    # ------------------------------------------------------------ public
    def search(
        self,
        query: str,
        *,
        k: int = 5,
        mode: Mode = "dense",
        year_from: int | None = None,
        year_to: int | None = None,
        min_score: float | None = None,
    ) -> list[SearchHit]:
        floor = self.min_score if min_score is None else min_score
        mask = self._year_mask(year_from, year_to)

        if mode == "bm25":
            return self._collapse(self._bm25_scores(query), mask, k)

        dense = self._dense_scores(query)
        if mode == "dense":
            hits = self._collapse(dense, mask, k)
            return [h for h in hits if h.score >= floor]

        if mode == "hybrid":
            d_hits = self._collapse(dense, mask, _POOL)
            b_hits = self._collapse(self._bm25_scores(query), mask, _POOL)
            fused: dict[str, float] = {}
            for ranked in (d_hits, b_hits):
                for rank, h in enumerate(ranked):
                    fused[h.movie_id] = fused.get(h.movie_id, 0.0) + 1.0 / (_RRF_K + rank + 1)
            dense_by_id = {h.movie_id: h for h in d_hits}
            order = sorted(fused, key=fused.get, reverse=True)
            # report the dense hit (its similarity + citation) and keep the min_score floor on it,
            # so a query that matches nothing semantically still returns nothing
            out = [dense_by_id[m] for m in order
                   if m in dense_by_id and dense_by_id[m].score >= floor]
            return out[:k]

        raise ValueError(f"unknown mode {mode!r}")

    def get_movie(self, movie_id: str) -> Movie | None:
        return self.movies.get(movie_id)

    # ------------------------------------------------------------ internals
    def _dense_scores(self, query: str) -> np.ndarray:
        if self.embedder is None:
            raise RuntimeError("dense search needs an embedder")
        q = self.embedder.embed_queries([query])[0]
        return self.store.scores(q)

    def _bm25_scores(self, query: str) -> np.ndarray:
        if self.bm25 is None:
            raise RuntimeError("BM25 index was not built")
        return self.bm25.scores(query)

    def _year_mask(self, year_from: int | None, year_to: int | None) -> np.ndarray:
        mask = np.ones(len(self._chunk_year), dtype=bool)
        if year_from is not None:
            mask &= self._chunk_year >= year_from
        if year_to is not None:
            mask &= self._chunk_year <= year_to
        return mask

    def _collapse(self, scores: np.ndarray, mask: np.ndarray, k: int) -> list[SearchHit]:
        """Top chunks -> best chunk per movie -> top-k movies."""
        scores = np.where(mask, scores, -np.inf)
        pool = min(len(scores), max(_POOL, k * 20))
        top = np.argpartition(-scores, pool - 1)[:pool] if pool < len(scores) else np.arange(len(scores))
        top = top[np.argsort(-scores[top], kind="stable")]

        hits: list[SearchHit] = []
        seen: set[str] = set()
        for i in top:
            if not np.isfinite(scores[i]):
                break
            mid = str(self._chunk_movie[i])
            if mid in seen:
                continue
            seen.add(mid)
            m, c = self.movies[mid], self.store.chunks[i]
            hits.append(
                SearchHit(
                    movie_id=mid,
                    title=m.title,
                    year=m.year,
                    score=round(float(scores[i]), 4),
                    chunk_id=c.chunk_id,
                    snippet=c.text[:300],
                )
            )
            if len(hits) == k:
                break
        return hits
