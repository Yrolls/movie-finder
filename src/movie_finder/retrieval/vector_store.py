"""A minimal in-memory vector store: a float32 matrix + the chunks it was built from.

Saved as two files in data/index/:  vectors.npy  and  chunks.jsonl (+ meta.json).
Cosine similarity == dot product because every vector is unit-length.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..schemas import Chunk


class VectorStore:
    def __init__(self, vectors: np.ndarray, chunks: list[Chunk], meta: dict | None = None):
        if len(vectors) != len(chunks):
            raise ValueError(f"{len(vectors)} vectors but {len(chunks)} chunks")
        self.vectors = vectors.astype(np.float32)
        self.chunks = chunks
        self.meta = meta or {}

    def scores(self, query_vec: np.ndarray) -> np.ndarray:
        return self.vectors @ query_vec.astype(np.float32)

    # ------------------------------------------------------------ persistence
    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "vectors.npy", self.vectors)
        with (folder / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(c.model_dump_json() + "\n")
        (folder / "meta.json").write_text(json.dumps(self.meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, folder: Path) -> "VectorStore":
        if not (folder / "vectors.npy").exists():
            raise FileNotFoundError(f"no index in {folder} -- run `python scripts/build_index.py`")
        vectors = np.load(folder / "vectors.npy")
        with (folder / "chunks.jsonl").open(encoding="utf-8") as f:
            chunks = [Chunk.model_validate_json(line) for line in f if line.strip()]
        meta_path = folder / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        return cls(vectors, chunks, meta)
