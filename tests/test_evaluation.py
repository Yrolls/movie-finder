import json

import pytest

from movie_finder.evaluation import (
    abstain_rate,
    find_titles,
    load_gold,
    norm_title,
    rank_of,
    resolve_gold,
    retrieval_metrics,
)
from movie_finder.schemas import Movie

P = "a long enough plot " * 5
MOVIES = [
    Movie(movie_id="m00000", title="The Loop Day", year=1993, plot=P),
    Movie(movie_id="m00001", title="Remake", year=1990, plot=P),
    Movie(movie_id="m00002", title="Remake", year=2010, plot=P),
    Movie(movie_id="m00003", title="Twin", year=2001, plot=P, origin="American"),
    Movie(movie_id="m00004", title="Twin", year=2001, plot=P, origin="Bollywood"),
]


def test_norm_title():
    assert norm_title("The Loop-Day!") == "loop day"
    assert norm_title("Amélie") == "amelie"


def gold_file(tmp_path, rows):
    p = tmp_path / "q.jsonl"
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n// comment\n",
                 encoding="utf-8")
    return load_gold(p)


def test_resolve_gold(tmp_path):
    gold = gold_file(tmp_path, [
        {"query": "a", "title": "Loop Day", "year": 1994},          # 'The' + year off by one
        {"query": "b", "title": "Remake", "year": 2010},
        {"query": "c", "title": None},                             # unanswerable
        {"query": "d", "title": "Missing", "year": 2000},
        {"query": "e", "title": "Twin", "year": 2001},
    ])
    problems = resolve_gold(gold, MOVIES)
    assert [g.movie_id for g in gold] == ["m00000", "m00002", None, None, None]
    assert len(problems) == 2
    assert "not found" in problems[0] and "ambiguous" in problems[1]
    assert not gold[2].answerable


def test_find_titles():
    assert [m.movie_id for m in find_titles(MOVIES, "remake")] == ["m00001", "m00002"]


def test_metrics():
    ranks = [1, 3, None, 7]
    m = retrieval_metrics(ranks, ks=(1, 5, 10))
    assert m["recall@1"] == 0.25
    assert m["recall@5"] == 0.5
    assert m["recall@10"] == 0.75
    assert m["mrr@10"] == pytest.approx((1 + 1 / 3 + 1 / 7) / 4)
    assert rank_of("b", ["a", "b"]) == 2 and rank_of("z", ["a"]) is None
    assert abstain_rate([0, 2, 0, 1]) == 0.5


def test_also_accepts_remakes(tmp_path):
    gold = gold_file(tmp_path, [
        {"query": "a", "title": "Remake", "year": 2010, "also": [{"title": "Remake", "year": 1990}]},
    ])
    assert resolve_gold(gold, MOVIES) == []
    assert gold[0].accepted == ["m00002", "m00001"]
    assert rank_of(gold[0].accepted, ["m00009", "m00001"]) == 2      # the older version also counts


def test_exact_title_breaks_ties(tmp_path):
    movies = MOVIES + [
        Movie(movie_id="m00010", title="Contact", year=1997, plot=P),
        Movie(movie_id="m00011", title="The Contact", year=1997, plot=P),
    ]
    gold = gold_file(tmp_path, [{"query": "a", "title": "Contact", "year": 1997},
                                {"query": "b", "title": "The Contact", "year": 1997}])
    assert resolve_gold(gold, movies) == []
    assert [g.movie_id for g in gold] == ["m00010", "m00011"]


def test_rank_metrics():
    from movie_finder.evaluation import bootstrap_ci, median_rank, ndcg
    assert ndcg([1, None]) == 0.5
    assert ndcg([3], k=10) == pytest.approx(0.5)            # 1/log2(4)
    assert median_rank([1, 2, None]) == 2 and median_rank([1, None]) == 6
    lo, hi = bootstrap_ci([1.0] * 10 + [0.0] * 10)
    assert lo < 0.5 < hi and bootstrap_ci([1.0, 0.0]) == bootstrap_ci([1.0, 0.0])   # seeded
