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


def answer_with_sql(question: str, db: MovieDB, client=None) -> dict:
    """Returns {"sql", "columns", "rows"} or {"sql", "error"}; never raises."""
    messages = [{"role": "system", "content": SQL_PROMPT}, {"role": "user", "content": question}]
    sql, error = "", ""
    for attempt in range(2):                                   # first try + one repair
        try:
            sql = chat_json(messages, GeneratedSQL, client=client).sql
        except Exception as e:  # noqa: BLE001 -- model/proxy failure becomes a readable error
            return {"sql": sql, "error": f"could not generate SQL: {type(e).__name__}"}
        try:
            cols, rows = db.run(sql)
            return {"sql": sql, "columns": cols, "rows": rows}
        except (SQLRefused, sqlite3.Error) as e:
            error = f"{type(e).__name__}: {e}"
            messages += [
                {"role": "assistant", "content": f'{{"sql": "{sql}"}}'},
                {"role": "user", "content": f"That query failed ({error}). Fix it and reply with ONLY the JSON."},
            ]
    return {"sql": sql, "error": error}
