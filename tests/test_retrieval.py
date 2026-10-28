"""Retrieval tests with a deterministic fake embedder (bag-of-words hashed into a vector)."""
import numpy as np
import pytest

from movie_finder.retrieval.bm25 import BM25Index
from movie_finder.retrieval.vector_store import VectorStore
from tests.fixtures import make_retriever


@pytest.fixture(scope="module")
def retriever():
    return make_retriever()


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid"])
def test_finds_right_movie(retriever, mode):
    hits = retriever.search("man relives the same winter day again and again", k=3, mode=mode)
    assert hits[0].movie_id == "m00000"
    assert hits[0].chunk_id.startswith("m00000#")          # citation points to the right movie


def test_one_hit_per_movie(retriever):
    hits = retriever.search("ship ice cold", k=4)
    ids = [h.movie_id for h in hits]
    assert len(ids) == len(set(ids))


def test_year_filter(retriever):
    hits = retriever.search("ship ice cold", k=4, year_from=1990)
    assert all(h.year >= 1990 for h in hits)
    assert "m00003" not in [h.movie_id for h in hits]


def test_min_score_floor_drops_junk(retriever):
    assert retriever.search("quantum chromodynamics lattice", k=3, min_score=0.3) == []
    assert retriever.search("quantum chromodynamics lattice", k=3, mode="hybrid", min_score=0.3) == []


def test_store_roundtrip(tmp_path, retriever):
    retriever.store.save(tmp_path)
    loaded = VectorStore.load(tmp_path)
    assert loaded.chunks == retriever.store.chunks
    assert np.allclose(loaded.vectors, retriever.store.vectors)


def test_bm25_prefers_rare_terms():
    idx = BM25Index(["the iceberg sank", "the ship sailed", "the ship docked"])
    s = idx.scores("iceberg ship")
    assert s.argmax() == 0
