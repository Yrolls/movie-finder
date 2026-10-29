"""See what retrieval returns for a description -- the top-k movies per mode.

    python scripts/search.py "a guy keeps living the same day in a snowy town"
    python scripts/search.py "หนังที่มีตัวตลกหลอกเด็ก" --rewrite      # rewrite with the LLM first, like the app
    python scripts/search.py "..." -k 10 --modes hybrid
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from movie_finder.retrieval.loader import load_retriever  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("description")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--modes", nargs="+", default=["bm25", "dense", "hybrid"])
    ap.add_argument("--rewrite", action="store_true", help="rewrite as an English plot with the LLM first")
    args = ap.parse_args()

    query = args.description
    if args.rewrite:
        from movie_finder.retrieval.rewrite import rewrite_query

        query = rewrite_query(args.description)
        print(f"rewritten: {query}\n")

    r = load_retriever(min_score=0.0)
    for mode in args.modes:
        print(f"--- {mode}")
        for i, h in enumerate(r.search(query, k=args.k, mode=mode, min_score=0.0), 1):
            print(f"{i:2d}. {h.title} ({h.year})  score={h.score:.3f}  [{h.chunk_id}]")
            print(f"    {h.snippet[:140]}...")
        print()


if __name__ == "__main__":
    main()
