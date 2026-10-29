"""Replay the agent's plot searches from eval results, without the LLM, to separate two causes of a
miss: the agent's wording vs. the retrieval code.

For every search_movies call recorded in one or more results.json files, run the same query
through the retriever as the app does (RETRIEVAL_MODE, TOP_K... from .env) twice:
  - current hybrid fusion
  - legacy hybrid fusion (films found only by BM25 are dropped -- the behaviour before the fix)
and report whether the gold film is in the returned list.

    python scripts/replay_searches.py eval/results/results.json data/processed/old_results.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from movie_finder.config import EVAL_DIR, PROCESSED_DIR, get_settings  # noqa: E402
from movie_finder.data.corpus import load_movies  # noqa: E402
from movie_finder.evaluation import load_gold, resolve_gold  # noqa: E402
from movie_finder.retrieval import retriever as R  # noqa: E402
from movie_finder.retrieval.loader import load_retriever  # noqa: E402
from movie_finder.schemas import SearchMoviesArgs  # noqa: E402


def legacy_hybrid(r, query, k, floor, year_from, year_to):
    mask = r._year_mask(year_from, year_to)
    dense = r._dense_scores(query)
    d_hits = r._collapse(dense, mask, R._POOL)
    b_hits = r._collapse(r._bm25_scores(query), mask, R._POOL)
    fused = {}
    for ranked in (d_hits, b_hits):
        for rank, h in enumerate(ranked):
            fused[h.movie_id] = fused.get(h.movie_id, 0.0) + 1.0 / (R._RRF_K + rank + 1)
    by_id = {h.movie_id: h for h in d_hits}
    out = [by_id[m] for m in sorted(fused, key=fused.get, reverse=True) if m in by_id and by_id[m].score >= floor]
    return out[:k]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+", type=Path)
    ap.add_argument("--gold", type=Path, default=EVAL_DIR / "queries.jsonl")
    ap.add_argument("--min-score", type=float, nargs="+", default=None,
                    help="also replay with these floors, e.g. --min-score 0 0.74 0.76 0.78")
    args = ap.parse_args()

    s = get_settings()
    movies = load_movies(PROCESSED_DIR / "movies.jsonl")
    gold = {g.qid: g for g in load_gold(args.gold)}
    resolve_gold(list(gold.values()), movies)
    r = load_retriever(movies=movies)                  # min_score / rerank from .env, as the app runs
    print(f"mode={s.retrieval_mode} min_score={r.min_score} reranker={'on' if r.reranker else 'off'}\n")

    for path in args.results:
        rows = json.loads(path.read_text(encoding="utf-8"))["agent"]["rows"]
        n = cur_ok = leg_ok = 0
        print(f"=== {path}")
        for row in rows:
            g = gold.get(row["id"])
            if g is None or not g.answerable:
                continue
            for raw in row["searches"]:
                a = json.loads(raw)
                if a.pop("within_database_results", False):
                    continue
                q = SearchMoviesArgs.model_validate(a)
                cur = r.search_many(q.all_queries, k=q.k, mode=s.retrieval_mode,
                                    year_from=q.year_from, year_to=q.year_to)
                leg = legacy_hybrid(r, q.all_queries[0], q.k, r.min_score, q.year_from, q.year_to)
                c = any(h.movie_id in g.accepted for h in cur)
                lg = any(h.movie_id in g.accepted for h in leg)
                n, cur_ok, leg_ok = n + 1, cur_ok + c, leg_ok + lg
                if c != lg:
                    print(f"  {row['id']}: current={'hit' if c else 'MISS'} legacy={'hit' if lg else 'MISS'}  "
                          f"{q.all_queries[0][:90]}")
                    print(f"      current top: {[h.title for h in cur]}")
                    print(f"      legacy  top: {[h.title for h in leg]}")
        print(f"  searches with the gold film returned: current {cur_ok}/{n}, legacy {leg_ok}/{n}")
        for floor in args.min_score or []:
            ok, miss = 0, []
            for row in rows:
                g = gold.get(row["id"])
                if g is None or not g.answerable:
                    continue
                for raw in row["searches"]:
                    a = json.loads(raw)
                    if a.pop("within_database_results", False):
                        continue
                    q = SearchMoviesArgs.model_validate(a)
                    hits = r.search_many(q.all_queries, k=q.k, mode=s.retrieval_mode, min_score=floor,
                                         year_from=q.year_from, year_to=q.year_to)
                    if any(h.movie_id in g.accepted for h in hits):
                        ok += 1
                    else:
                        miss.append(row["id"])
            print(f"  min_score {floor:.2f}: {ok}/{n}  misses: {' '.join(dict.fromkeys(miss))}")
        print()


if __name__ == "__main__":
    main()
