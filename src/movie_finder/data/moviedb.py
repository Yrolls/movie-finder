"""Read-only SQL over the movie metadata (year, origin, genre, director, cast).

Plot search can't use what people often DO remember about a film -- an actor, the director,
the decade, the country. Those are exact facts, so they belong in a database query, not in an
embedding (Lecture 9: "after 2020 is a filter, not a vibe"; Lecture 11: text-to-SQL).

Defence in depth for model-written SQL (Lecture 11):
  1. read-only connection  -- PRAGMA query_only=ON: no statement can write, whatever it says
  2. single SELECT only    -- must start with SELECT/WITH, no ';'
  3. keyword denylist      -- insert/update/delete/drop/attach/pragma/... rejected anywhere
  4. table allow-list      -- only the `movies` table may be referenced
  5. no title filter       -- the user doesn't know the title; filtering on it would let the
                              model look up its own guess instead of the user's memory
  6. forced LIMIT + a time budget on execution
"""
from __future__ import annotations

import re
import sqlite3
import time
from collections.abc import Iterable

from ..schemas import Movie

SCHEMA_DDL = """CREATE TABLE movies (
  movie_id TEXT PRIMARY KEY,  -- e.g. 'm04161'
  title    TEXT,              -- do NOT filter on title (the user does not know it)
  year     INTEGER,           -- release year, 1980-2017
  origin   TEXT,              -- e.g. 'American', 'British', 'Japanese', 'South_Korean', 'Hong Kong', 'Bollywood', 'Tamil', 'Chinese'
  genre    TEXT,              -- free text, lowercase, e.g. 'science fiction', 'crime drama', 'anime'
  director TEXT,              -- comma-separated names
  actors   TEXT               -- comma-separated names of the main cast
);"""

MAX_ROWS = 2000          # all matches are kept for the plot-search pre-filter; only a sample is shown
TIME_BUDGET_S = 2.0

_BANNED = re.compile(r"(?is)\b(insert|update|delete|drop|alter|create|attach|detach|pragma|"
                     r"replace|vacuum|reindex|trigger|load_extension)\b")
_TABLES = re.compile(r"(?is)\b(?:from|join)\s+([a-z_][a-z0-9_]*)")
_WHERE = re.compile(r"(?is)\bwhere\b(.*?)(?:\border\s+by\b|\bgroup\s+by\b|\blimit\b|$)")
_LIMIT = re.compile(r"(?is)\blimit\s+\d+\s*$")


class SQLRefused(ValueError):
    """The generated SQL failed a guard; the message says which one."""


def clean_sql(text: str) -> str:
    """Strip ```sql fences, a leading 'sql' word and trailing ';' the model tends to add."""
    sql = text.strip()
    m = re.search(r"```(?:sql)?\s*(.*?)```", sql, flags=re.S | re.I)
    if m:
        sql = m.group(1)
    sql = re.sub(r"^\s*sql\b", "", sql.strip().strip("`"), flags=re.I)
    return sql.strip().rstrip(";").strip()


def guard_sql(sql: str) -> None:
    """Raise SQLRefused if the query is not a single, read-only SELECT over `movies`."""
    if not re.match(r"(?is)^\s*(select|with)\b", sql):
        raise SQLRefused("not a SELECT")
    if ";" in sql:
        raise SQLRefused("multiple statements")
    if m := _BANNED.search(sql):
        raise SQLRefused(f"banned keyword: {m.group(1).upper()}")
    tables = {t.lower() for t in _TABLES.findall(sql)}
    if tables - {"movies"}:
        raise SQLRefused(f"only the movies table is allowed, got {sorted(tables)}")
    if any(re.search(r"(?i)\btitle\b", w) for w in _WHERE.findall(sql)):
        raise SQLRefused("filtering on title is not allowed -- filter on what the user remembers")


def add_limit(sql: str, n: int = MAX_ROWS) -> str:
    return sql if _LIMIT.search(sql) else f"{sql} LIMIT {n}"


class MovieDB:
    def __init__(self, movies: Iterable[Movie]):
        self._db = sqlite3.connect(":memory:", check_same_thread=False)
        self._db.execute(SCHEMA_DDL)
        self._db.executemany(
            "INSERT INTO movies VALUES (?,?,?,?,?,?,?)",
            [(m.movie_id, m.title, m.year, m.origin, m.genre.lower(), _names(m.director), _names(m.cast))
             for m in movies],
        )
        self._db.commit()
        self._db.execute("PRAGMA query_only = ON")          # layer 1: the connection cannot write

    def run(self, sql: str) -> tuple[list[str], list[tuple]]:
        """Guard, bound and execute model-written SQL. Raises SQLRefused or sqlite3.Error."""
        sql = add_limit(clean_sql(sql))
        guard_sql(sql)
        deadline = time.monotonic() + TIME_BUDGET_S
        self._db.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
        try:
            cur = self._db.execute(sql)
            rows = cur.fetchmany(MAX_ROWS)
        finally:
            self._db.set_progress_handler(None, 0)
        cols = [d[0] for d in cur.description or []]
        return cols, rows


def _names(s: str) -> str:
    """Cast/director cells come comma- or newline-separated; store one comma-separated list."""
    parts = [p.strip() for p in re.split(r"[,\n]+", s or "") if p.strip()]
    return ", ".join(parts)
