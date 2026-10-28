"""Load, clean and chunk the movie-plot corpus.

Raw source: Kaggle "Wikipedia Movie Plots" (wiki_movie_plots_deduped.csv, CC BY-SA 4.0).
Processed form: one JSON object per line (Movie schema) in data/processed/movies.jsonl.
"""
from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from pathlib import Path

import pandas as pd

from ..schemas import Chunk, Movie

RAW_COLUMNS = {
    "Release Year": "year",
    "Title": "title",
    "Origin/Ethnicity": "origin",
    "Director": "director",
    "Cast": "cast",
    "Genre": "genre",
    "Wiki Page": "wiki_url",
    "Plot": "plot",
}

_CITATION_MARK = re.compile(r"\[\d+\]")      # Wikipedia footnote markers like [3]
_WS = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = _CITATION_MARK.sub("", str(text))
    return _WS.sub(" ", text).strip()


def _clean_field(v) -> str:
    v = "" if pd.isna(v) else str(v).strip()
    return "" if v.lower() in {"unknown", "nan"} else v


def prepare_movies(
    df: pd.DataFrame,
    *,
    min_year: int = 1980,
    max_year: int = 2100,
    origins: Iterable[str] | None = None,
    min_plot_words: int = 60,
) -> list[Movie]:
    """Filter + clean the raw Kaggle frame and assign stable ids (sorted by year, title)."""
    missing = set(RAW_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"raw CSV is missing columns: {sorted(missing)}")

    df = df.rename(columns=RAW_COLUMNS)[list(RAW_COLUMNS.values())].copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year", "title", "plot"])
    df["year"] = df["year"].astype(int)
    df = df[(df["year"] >= min_year) & (df["year"] <= max_year)]
    if origins:
        df = df[df["origin"].isin(set(origins))]

    df["plot"] = df["plot"].map(clean_text)
    df = df[df["plot"].str.split().str.len() >= min_plot_words]
    df = df.drop_duplicates(subset=["title", "year"])
    df = df.sort_values(["year", "title"], kind="stable").reset_index(drop=True)

    movies = []
    for i, row in df.iterrows():
        movies.append(
            Movie(
                movie_id=f"m{i:05d}",
                title=str(row["title"]).strip(),
                year=int(row["year"]),
                origin=_clean_field(row["origin"]),
                genre=_clean_field(row["genre"]),
                director=_clean_field(row["director"]),
                cast=_clean_field(row["cast"]),
                wiki_url=_clean_field(row["wiki_url"]),
                plot=row["plot"],
            )
        )
    return movies


# ---------------------------------------------------------------- jsonl io
def save_movies(movies: Iterable[Movie], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for m in movies:
            f.write(m.model_dump_json() + "\n")
            n += 1
    return n


def load_movies(path: Path) -> list[Movie]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run `python scripts/fetch_data.py` first")
    with path.open(encoding="utf-8") as f:
        return [Movie.model_validate(json.loads(line)) for line in f if line.strip()]


# ---------------------------------------------------------------- chunking
def chunk_movie(movie: Movie, *, words: int, overlap: int) -> Iterator[Chunk]:
    """Split a plot into overlapping word windows.

    The title is deliberately NOT put into the chunk text: users can't remember it, so it
    must not be what retrieval matches on.
    """
    if overlap >= words:
        raise ValueError("overlap must be smaller than words")
    tokens = movie.plot.split()
    step = words - overlap
    starts = range(0, max(len(tokens) - overlap, 1), step)
    for n, start in enumerate(starts):
        piece = tokens[start : start + words]
        if not piece:
            break
        yield Chunk(chunk_id=f"{movie.movie_id}#{n}", movie_id=movie.movie_id, text=" ".join(piece))


def chunk_corpus(movies: Iterable[Movie], *, words: int, overlap: int) -> list[Chunk]:
    return [c for m in movies for c in chunk_movie(m, words=words, overlap=overlap)]
