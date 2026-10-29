import json

from movie_finder.agent.sql_tool import answer_with_sql
from movie_finder.data.moviedb import MovieDB
from tests.fixtures import MOVIES

GOOD = json.dumps({"sql": "SELECT movie_id, title, year, director, actors FROM movies WHERE actors LIKE '%Murray%'"})


def test_generates_and_runs(fake_client):
    client = fake_client([GOOD])
    out = answer_with_sql("a movie with Bill Murray", MovieDB(MOVIES), client=client)
    assert [r[0] for r in out["rows"]] == ["m00000"]
    assert "CREATE TABLE movies" in client.calls[0]["messages"][0]["content"]   # literal DDL in prompt
    assert client.calls[0]["temperature"] == 0.0


def test_error_is_fed_back_for_one_retry(fake_client):
    bad = json.dumps({"sql": "SELECT nope FROM movies"})
    client = fake_client([bad, GOOD])
    out = answer_with_sql("a movie with Bill Murray", MovieDB(MOVIES), client=client)
    assert out["rows"] and "no such column" in client.calls[1]["messages"][-1]["content"]


def test_refused_twice_returns_error(fake_client):
    title = json.dumps({"sql": "SELECT * FROM movies WHERE title = 'Loop Day'"})
    out = answer_with_sql("loop day", MovieDB(MOVIES), client=fake_client([title, title]))
    assert "title" in out["error"]
