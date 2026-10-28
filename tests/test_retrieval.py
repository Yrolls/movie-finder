"""Retrieval tests with a deterministic fake embedder (bag-of-words hashed into a vector)."""
import zlib

import numpy as np
import pytest

from movie_finder.data.corpus import chunk_corpus
from movie_finder.retrieval.bm25 import BM25Index, tokenize
from movie_finder.retrieval.retriever import Retriever
from movie_finder.retrieval.vector_store import VectorStore
from movie_finder.schemas import Movie

DIM = 256


class HashEmbedder:
    def _vec(self, text):
        v = np.zeros(DIM, dtype=np.float32)
        for t in tokenize(text):
            v[zlib.crc32(t.encode()) % DIM] += 1
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_queries(self, texts):
        return np.stack([self._vec(t) for t in texts])

    embed_passages = embed_queries


# invented mini-corpus
MOVIES = [
    Movie(movie_id="m00000", title="Loop Day", year=1993,
          plot="A grumpy weatherman wakes up every morning to relive the same winter day in a small town "
               "until he learns to become a better person."),
    Movie(movie_id="m00001", title="Dream Heist", year=2010,
          plot="A thief enters people's dreams to steal secrets, building layers of dreams within dreams "
               "while his dead wife haunts his subconscious."),
    Movie(movie_id="m00002", title="Ship of Ice", year=1997,
          plot="A poor artist and a rich young woman fall in love aboard a huge ocean liner that strikes "
               "an iceberg and sinks in the cold Atlantic."),
    Movie(movie_id="m00003", title="Ship of Ice 2", year=1950,
          plot="An old cargo ship gets trapped in arctic ice and the crew must survive the cold winter."),
]


@pytest.fixture(scope="module")
def retriever():
    chunks = chunk_corpus(MOVIES, words=12, overlap=4)
    emb = HashEmbedder()
    store = VectorStore(emb.embed_passages([c.text for c in chunks]), chunks)
    return Retriever(MOVIES, store, emb, min_score=0.0)


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
