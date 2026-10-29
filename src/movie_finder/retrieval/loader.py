"""Build a ready-to-use Retriever from data/ on disk (shared by the API and the eval)."""
from __future__ import annotations

from pathlib import Path

from ..config import INDEX_DIR, PROCESSED_DIR, get_settings
from ..data.corpus import load_movies
from ..schemas import Movie
from .embedder import SentenceTransformerEmbedder
from .retriever import Retriever
from .vector_store import VectorStore

_EMBEDDERS: dict[str, SentenceTransformerEmbedder] = {}
_RERANKERS: dict = {}


def get_reranker(model_name: str):
    if not model_name:
        return None
    if model_name not in _RERANKERS:
        from .reranker import CrossEncoderReranker
        _RERANKERS[model_name] = CrossEncoderReranker(model_name)
    return _RERANKERS[model_name]


def get_embedder(model_name: str) -> SentenceTransformerEmbedder:
    if model_name not in _EMBEDDERS:                     # loading the model is slow; do it once
        _EMBEDDERS[model_name] = SentenceTransformerEmbedder(model_name)
    return _EMBEDDERS[model_name]


def load_retriever(
    index_dir: Path = INDEX_DIR,
    *,
    movies: list[Movie] | None = None,
    min_score: float | None = None,
    rerank_model: str | None = None,
) -> Retriever:
    s = get_settings()
    movies = movies if movies is not None else load_movies(PROCESSED_DIR / "movies.jsonl")
    store = VectorStore.load(index_dir)
    model = store.meta.get("embed_model", s.embed_model)  # query with the model the index was built with
    return Retriever(
        movies,
        store,
        get_embedder(model),
        min_score=s.min_score if min_score is None else min_score,
        reranker=get_reranker(s.rerank_model if rerank_model is None else rerank_model),
    )
