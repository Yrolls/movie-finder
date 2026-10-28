import pandas as pd
import pytest

from movie_finder.data.corpus import (
    chunk_movie,
    clean_text,
    load_movies,
    prepare_movies,
    save_movies,
)
from movie_finder.schemas import Movie

LONG_PLOT = " ".join(f"w{i}" for i in range(130))


def raw_frame():
    # invented rows in the Kaggle column layout
    return pd.DataFrame(
        {
            "Release Year": [1999, 1975, 1999, 1960, 2001],
            "Title": ["Beta", "Alpha", "Beta", "Old", "Short"],
            "Origin/Ethnicity": ["American", "British", "American", "American", "Japanese"],
            "Director": ["A", "unknown", "A", "B", "C"],
            "Cast": ["X", "Y", "X", "Z", "W"],
            "Genre": ["drama", "comedy", "drama", "war", "unknown"],
            "Wiki Page": ["u1", "u2", "u1", "u3", "u4"],
            "Plot": [LONG_PLOT + " [3]", LONG_PLOT, LONG_PLOT, LONG_PLOT, "too short"],
        }
    )


def test_prepare_filters_and_cleans():
    movies = prepare_movies(raw_frame(), min_year=1970)
    titles = [m.title for m in movies]
    assert titles == ["Alpha", "Beta"]                 # sorted, deduped, old + short dropped
    assert movies[0].movie_id == "m00000"
    assert movies[0].director == ""                    # "unknown" -> ""
    assert "[3]" not in movies[1].plot


def test_prepare_origin_filter():
    movies = prepare_movies(raw_frame(), min_year=1970, origins=["British"])
    assert [m.title for m in movies] == ["Alpha"]


def test_prepare_rejects_wrong_csv():
    with pytest.raises(ValueError, match="missing columns"):
        prepare_movies(pd.DataFrame({"foo": [1]}))


def test_clean_text():
    assert clean_text("He  runs.[12]\nShe   hides.") == "He runs. She hides."


def test_chunking_overlap_and_coverage():
    m = Movie(movie_id="m00001", title="T", year=2000, plot=LONG_PLOT)
    chunks = list(chunk_movie(m, words=50, overlap=10))
    assert [c.chunk_id for c in chunks] == ["m00001#0", "m00001#1", "m00001#2"]
    assert chunks[1].text.split()[0] == "w40"           # step = 40
    assert chunks[-1].text.split()[-1] == "w129"        # nothing lost at the end
    assert all("T" not in c.text.split() for c in chunks)  # title never leaks into text


def test_short_plot_single_chunk():
    m = Movie(movie_id="m00002", title="T", year=2000, plot="one two three")
    assert len(list(chunk_movie(m, words=50, overlap=10))) == 1


def test_jsonl_roundtrip(tmp_path):
    movies = prepare_movies(raw_frame(), min_year=1970)
    p = tmp_path / "movies.jsonl"
    assert save_movies(movies, p) == 2
    assert load_movies(p) == movies
