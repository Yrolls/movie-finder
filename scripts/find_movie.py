"""Look up movies in the corpus by (part of) the title -- for writing eval/queries.jsonl.

    python scripts/find_movie.py "groundhog"
    python scripts/find_movie.py "taken" --plot          # also print the start of the plot
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from movie_finder.config import PROCESSED_DIR  # noqa: E402
from movie_finder.data.corpus import load_movies  # noqa: E402
from movie_finder.evaluation import find_titles  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("title")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    hits = find_titles(load_movies(PROCESSED_DIR / "movies.jsonl"), args.title)
    if not hits:
        print("no title contains that text (the corpus covers 1980-2017)")
    for m in hits:
        print(f'{m.movie_id}  {{"title": "{m.title}", "year": {m.year}}}  [{m.origin}]')
        if args.plot:
            print("    " + " ".join(m.plot.split()[:60]) + " ...\n")


if __name__ == "__main__":
    main()
