"""query_movie_database: natural-language question -> guarded SQL -> rows (Lecture 11 pattern).

The agent passes a QUESTION, not SQL. This module writes the SQL with the LLM from the literal
DDL + few-shot examples (one per capability, Lecture 9), runs it through the guards in
data/moviedb.py, and on a SQLite or guard error feeds the error back for ONE retry.
"""
from __future__ import annotations

import sqlite3

from pydantic import BaseModel, Field

from ..data.moviedb import SCHEMA_DDL, MovieDB, SQLRefused
from ..llm import chat_json

SQL_PROMPT = f"""Write ONE SQLite SELECT that answers the question using only this table:

{SCHEMA_DDL}

Rules:
- Use LIKE '%name%' for director, actors, genre and origin (names are comma-separated, spelling varies).
- Only filter on facts the question states: people, years or decades, country, genre.
  Never filter on title. Years in memories are fuzzy: widen a decade by 2 years on each side.
- genre is often empty or inconsistent: filter on it only when the question gives no person,
  year or country.
- Always SELECT movie_id, title, year, director, actors.
- Reply with ONLY: {{"sql": "<the query>"}}

Examples:
Q: a movie with Tom Hanks
A: {{"sql": "SELECT movie_id, title, year, director, actors FROM movies WHERE actors LIKE '%Tom Hanks%'"}}
Q: a Spielberg film from the 90s
A: {{"sql": "SELECT movie_id, title, year, director, actors FROM movies WHERE director LIKE '%Spielberg%' AND year BETWEEN 1988 AND 2001"}}
Q: a Korean movie
A: {{"sql": "SELECT movie_id, title, year, director, actors FROM movies WHERE origin LIKE '%Korean%'"}}
Q: an animated movie
A: {{"sql": "SELECT movie_id, title, year, director, actors FROM movies WHERE genre LIKE '%anim%'"}}
Q: Leonardo DiCaprio and Kate Winslet together
A: {{"sql": "SELECT movie_id, title, year, director, actors FROM movies WHERE actors LIKE '%DiCaprio%' AND actors LIKE '%Winslet%'"}}
Q: a Japanese anime directed by Miyazaki
A: {{"sql": "SELECT movie_id, title, year, director, actors FROM movies WHERE origin LIKE '%Japanese%' AND director LIKE '%Miyazaki%'"}}"""


class GeneratedSQL(BaseModel):
    sql: str = Field(min_length=10, max_length=1000)


RELAX = ("That query returned 0 rows -- the conditions are too strict, or the user misremembered "
         "one. Drop the least certain condition (genre first, then the year range) and reply with "
         "ONLY the JSON.")


def answer_with_sql(question: str, db: MovieDB, client=None) -> dict:
    """Returns {"sql", "columns", "rows"[, "relaxed_from"]} or {"sql", "error"}; never raises.

    One extra turn is allowed, used either to repair a failing query (error fed back) or to
    relax a query that matched nothing (e.g. an over-specific genre filter).
    """
    messages = [{"role": "system", "content": SQL_PROMPT}, {"role": "user", "content": question}]
    sql, error, empty = "", "", None
    for attempt in range(2):                                   # first try + one repair/relax
        try:
            sql = chat_json(messages, GeneratedSQL, client=client).sql
        except Exception as e:  # noqa: BLE001 -- model/proxy failure becomes a readable error
            return {"sql": sql, "error": f"could not generate SQL: {type(e).__name__}"}
        try:
            cols, rows = db.run(sql)
        except (SQLRefused, sqlite3.Error) as e:
            error = f"{type(e).__name__}: {e}"
            feedback = f"That query failed ({error}). Fix it and reply with ONLY the JSON."
        else:
            if rows or attempt == 1:
                out = {"sql": sql, "columns": cols, "rows": rows}
                if empty:
                    out["relaxed_from"] = empty
                return out
            empty, feedback = sql, RELAX
        messages += [{"role": "assistant", "content": f'{{"sql": "{sql}"}}'},
                     {"role": "user", "content": feedback}]
    return {"sql": sql, "error": error}
