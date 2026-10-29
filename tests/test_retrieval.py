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


def test_search_many_fuses_phrasings(retriever):
    # neither phrasing alone is great; together they agree on the right movie
    hits = retriever.search_many(["man stuck in a small town in winter",
                                  "weatherman relives the same day"], k=3)
    assert hits[0].movie_id == "m00000"
    assert len({h.movie_id for h in hits}) == len(hits)


def test_search_many_single_query_equals_search(retriever):
    a = retriever.search("ship ice cold", k=3)
    b = retriever.search_many(["ship ice cold", "ship ice cold "], k=3)   # duplicate collapses
    assert [h.movie_id for h in a] == [h.movie_id for h in b]


def test_hybrid_keeps_bm25_only_hits():
    """A film BM25 finds but dense misses must survive fusion (regression: eval q15/q24/q30)."""
    from movie_finder.retrieval.retriever import Retriever
    from tests.fixtures import MOVIES, HashEmbedder
    from movie_finder.data.corpus import chunk_corpus

    class BlindDense(HashEmbedder):          # dense "never" ranks m00002 (all its chunks score -1)
        pass

    chunks = chunk_corpus(MOVIES, words=12, overlap=4)
    emb = BlindDense()
    vecs = emb.embed_passages([c.text for c in chunks])
    for i, c in enumerate(chunks):
        if c.movie_id == "m00002":
            vecs[i] = -vecs[i]
    r = Retriever(MOVIES, VectorStore(vecs, chunks), emb, min_score=-1.0)
    import movie_finder.retrieval.retriever as R
    old = R._POOL
    R._POOL = 2                               # dense pool too small to contain m00002
    try:
        hits = r.search("iceberg ocean liner sinks atlantic", k=3, mode="hybrid")
    finally:
        R._POOL = old
    assert "m00002" in [h.movie_id for h in hits]


class KeywordReranker:
    """Fake cross-encoder: scores a passage by how many query words it contains."""
    def scores(self, query, passages):
        q = set(query.lower().split())
        return [float(len(q & set(p.lower().split()))) for p in passages]


def test_rerank_reorders_but_keeps_similarity(retriever):
    base = retriever.search("ship cold winter", k=4, mode="bm25", rerank=False)
    retriever.reranker = KeywordReranker()
    try:
        rr = retriever.search("ship cold winter", k=4, mode="bm25")          # on by default now
        off = retriever.search("ship cold winter", k=4, mode="bm25", rerank=False)
    finally:
        retriever.reranker = None
    assert {h.movie_id for h in rr} == {h.movie_id for h in base}
    assert [h.movie_id for h in off] == [h.movie_id for h in base]
    by_id = {h.movie_id: h.score for h in base}
    assert all(h.score == by_id[h.movie_id] for h in rr)                     # order changes only


def test_rerank_without_model_is_an_error(retriever):
    with pytest.raises(RuntimeError):
        retriever.search("ship", k=2, rerank=True)


def test_model_specific_prefixes():
    from movie_finder.retrieval.embedder import BGE_QUERY, prefixes_for
    assert prefixes_for("intfloat/multilingual-e5-small") == ("query: ", "passage: ")
    assert prefixes_for("BAAI/bge-small-en-v1.5") == (BGE_QUERY, "")
    assert prefixes_for("sentence-transformers/all-MiniLM-L6-v2") == ("", "")
