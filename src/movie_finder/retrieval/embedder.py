"""Text -> unit-length vectors.

Default model: intfloat/multilingual-e5-small (local, ~470 MB, CPU-friendly). It is multilingual,
so a Thai description can match an English plot. Because the agent rewrites every description
into English first, an English model (e.g. BAAI/bge-small-en-v1.5) is also evaluated. E5 models expect "query: " / "passage: "
prefixes; we add them here so the rest of the code never has to remember.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    def embed_queries(self, texts: list[str]) -> np.ndarray: ...
    def embed_passages(self, texts: list[str]) -> np.ndarray: ...


BGE_QUERY = "Represent this sentence for searching relevant passages: "


def prefixes_for(model_name: str) -> tuple[str, str]:
    """(query prefix, passage prefix) each model family was trained with."""
    name = model_name.lower()
    if "e5" in name:
        return "query: ", "passage: "
    if "bge" in name and "-en" in name:            # English BGE v1.5: instruction on queries only
        return BGE_QUERY, ""
    return "", ""


def _normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.clip(norms, 1e-12, None)).astype(np.float32)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, batch_size: int = 64):
        from sentence_transformers import SentenceTransformer  # heavy import, load lazily

        self.model_name = model_name
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name)
        self._query_prefix, self._passage_prefix = prefixes_for(model_name)

    def _encode(self, texts: list[str], prefix: str, show_progress: bool) -> np.ndarray:
        if prefix:
            texts = [prefix + t for t in texts]
        vecs = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return _normalize(np.asarray(vecs))

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, self._query_prefix, show_progress=False)

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, self._passage_prefix, show_progress=len(texts) > 200)
