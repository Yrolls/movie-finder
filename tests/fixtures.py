"""Tiny invented corpus + deterministic embedder shared by retrieval/agent/eval tests."""
import zlib

import numpy as np

from movie_finder.data.corpus import chunk_corpus
from movie_finder.retrieval.bm25 import tokenize
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


MOVIES = [
    Movie(movie_id="m00000", title="Loop Day", year=1993, origin="American", genre="Comedy",
          director="Harold Ramis", cast="Bill Murray, Andie MacDowell",
          plot="A grumpy weatherman wakes up every morning to relive the same winter day in a small town "
               "until he learns to become a better person."),
    Movie(movie_id="m00001", title="Dream Heist", year=2010, origin="American",
          genre="science fiction", director="Christopher Nolan", cast="Leonardo DiCaprio\nTom Hardy",
          plot="A thief enters people's dreams to steal secrets, building layers of dreams within dreams "
               "while his dead wife haunts his subconscious."),
    Movie(movie_id="m00002", title="Ship of Ice", year=1997, origin="American",
          director="James Cameron", cast="Leonardo DiCaprio, Kate Winslet",
          plot="A poor artist and a rich young woman fall in love aboard a huge ocean liner that strikes "
               "an iceberg and sinks in the cold Atlantic."),
    Movie(movie_id="m00003", title="Ship of Ice 2", year=1950,
          plot="An old cargo ship gets trapped in arctic ice and the crew must survive the cold winter."),
]


def make_retriever(min_score=0.0):
    chunks = chunk_corpus(MOVIES, words=12, overlap=4)
    emb = HashEmbedder()
    store = VectorStore(emb.embed_passages([c.text for c in chunks]), chunks)
    return Retriever(MOVIES, store, emb, min_score=min_score)
