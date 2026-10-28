"""Download the Wikipedia Movie Plots dataset and write the cleaned corpus.

    python scripts/fetch_data.py                       # download via kagglehub, then prepare
    python scripts/fetch_data.py --csv path/to.csv     # use a CSV you downloaded by hand
    python scripts/fetch_data.py --min-year 1990 --origins American British

Output: data/processed/movies.jsonl   (one Movie per line)
Source: https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots  (CC BY-SA 4.0)
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from movie_finder.config import PROCESSED_DIR, RAW_DIR  # noqa: E402
from movie_finder.data.corpus import prepare_movies, save_movies  # noqa: E402

KAGGLE_DATASET = "jrobischon/wikipedia-movie-plots"
CSV_NAME = "wiki_movie_plots_deduped.csv"
RAW_CSV = RAW_DIR / CSV_NAME
OUT = PROCESSED_DIR / "movies.jsonl"


def download() -> Path:
    if RAW_CSV.exists():
        print(f"[fetch] using cached {RAW_CSV}")
        return RAW_CSV
    try:
        import kagglehub
    except ImportError:
        sys.exit("kagglehub is not installed -- `pip install kagglehub`, or pass --csv")
    print(f"[fetch] downloading {KAGGLE_DATASET} via kagglehub ...")
    folder = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    src = next(folder.rglob(CSV_NAME), None)
    if src is None:
        sys.exit(f"{CSV_NAME} not found in {folder}")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, RAW_CSV)
    return RAW_CSV


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, help="local copy of wiki_movie_plots_deduped.csv")
    ap.add_argument("--min-year", type=int, default=1980)
    ap.add_argument("--max-year", type=int, default=2100)
    ap.add_argument("--origins", nargs="*", help="e.g. American British Japanese (default: all)")
    ap.add_argument("--min-plot-words", type=int, default=60)
    args = ap.parse_args()

    csv_path = args.csv or download()
    df = pd.read_csv(csv_path)
    print(f"[fetch] raw rows: {len(df):,}")

    movies = prepare_movies(
        df,
        min_year=args.min_year,
        max_year=args.max_year,
        origins=args.origins,
        min_plot_words=args.min_plot_words,
    )
    n = save_movies(movies, OUT)
    years = [m.year for m in movies]
    print(f"[fetch] kept {n:,} movies ({min(years)}-{max(years)}) -> {OUT}")


if __name__ == "__main__":
    main()
