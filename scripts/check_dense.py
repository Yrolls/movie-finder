"""Sanity check for the dense index: can it find a movie from a passage of its own plot?

    python scripts/check_dense.py            # 100 random chunks, fixed seed

If this is near 100%, the embed -> index -> search pipeline is wired correctly and low eval scores
come from the queries being hard (vague / other language), not from a bug.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from movie_finder.retrieval.loader import load_retriever  # noqa: E402


def main(n: int = 100, words: int = 40) -> None:
    r = load_retriever(min_score=0.0)
    rng = random.Random(0)
    sample = rng.sample(r.store.chunks, n)
    for mode in ("bm25", "dense", "hybrid"):
        top1 = top5 = 0
        for c in sample:
            query = " ".join(c.text.split()[:words])
            ids = [h.movie_id for h in r.search(query, k=5, mode=mode, min_score=0.0)]
            top1 += ids[:1] == [c.movie_id]
            top5 += c.movie_id in ids
        print(f"{mode:7s}  own-movie recall@1 = {top1 / n:.0%}   recall@5 = {top5 / n:.0%}")


if __name__ == "__main__":
    main()
