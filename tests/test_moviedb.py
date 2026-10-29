import sqlite3

import pytest

from movie_finder.data.moviedb import MovieDB, SQLRefused, add_limit, clean_sql, guard_sql
from tests.fixtures import MOVIES

COLS = "SELECT movie_id, title, year, director, actors FROM movies"


@pytest.fixture(scope="module")
def db():
    return MovieDB(MOVIES)


def test_actor_and_decade_query(db):
    cols, rows = db.run(f"{COLS} WHERE actors LIKE '%DiCaprio%' AND year BETWEEN 1990 AND 2001")
    assert cols[:3] == ["movie_id", "title", "year"]
    assert [r[0] for r in rows] == ["m00002"]


def test_newline_separated_cast_is_normalised(db):
    _, rows = db.run(f"{COLS} WHERE actors LIKE '%Tom Hardy%'")
    assert rows[0][4] == "Leonardo DiCaprio, Tom Hardy"


@pytest.mark.parametrize("sql, reason", [
    ("DELETE FROM movies", "not a SELECT"),
    ("SELECT 1; DROP TABLE movies", "multiple statements"),
    ("SELECT * FROM movies -- then DROP TABLE movies", "banned keyword"),
    ("SELECT * FROM sqlite_master", "only the movies table"),
    ("SELECT m.title FROM movies m JOIN secrets s ON 1=1", "only the movies table"),
    (f"{COLS} WHERE title LIKE '%Titanic%'", "title"),
    (f"{COLS} WHERE year > 2000 AND lower(title) = 'x' ORDER BY year", "title"),
])
def test_guard_refuses(sql, reason):
    with pytest.raises(SQLRefused, match=reason):
        guard_sql(sql)


def test_guard_allows_order_by_title_and_with():
    guard_sql(f"{COLS} WHERE year > 2000 ORDER BY title")
    guard_sql("WITH x AS (SELECT * FROM movies) SELECT * FROM movies")


def test_connection_is_read_only_even_without_the_guard(db):
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        db._db.execute("DELETE FROM movies")


def test_clean_and_limit():
    assert clean_sql("```sql\nSELECT 1 FROM movies;\n```") == "SELECT 1 FROM movies"
    assert add_limit("SELECT 1 FROM movies").endswith("LIMIT 2000")
    assert add_limit("SELECT 1 FROM movies LIMIT 3") == "SELECT 1 FROM movies LIMIT 3"


def test_run_applies_limit(db):
    _, rows = db.run(f"{COLS}")
    assert len(rows) == len(MOVIES)
