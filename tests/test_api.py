"""API tests (skipped where FastAPI isn't installed). The LLM is replaced by a scripted fake."""
import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")
from fastapi.testclient import TestClient  # noqa: E402

from movie_finder import llm  # noqa: E402
from movie_finder.api.app import create_app  # noqa: E402
from tests.conftest import FakeClient, tool_call  # noqa: E402
from tests.fixtures import make_retriever  # noqa: E402

FINAL = json.dumps({"status": "found", "candidates": [
    {"movie_id": "m00000", "title": "x", "year": 1, "confidence": 0.9,
     "evidence_chunk_id": "m00000#0", "why": "same day repeats"}]})


@pytest.fixture
def client(monkeypatch):
    fake = FakeClient([tool_call("search_movies", json.dumps({"query": "same winter day again"})), FINAL])
    monkeypatch.setattr(llm, "get_client", lambda: fake)
    with TestClient(create_app(make_retriever())) as c:
        yield c


def test_health_and_index(client):
    assert client.get("/health").json()["movies"] == 4
    assert "หนังเรื่องนั้นชื่ออะไรนะ?" in client.get("/").text


def test_search_validates_input(client):
    assert client.post("/search", json={"query": "same winter day", "k": 2}).status_code == 200
    assert client.post("/search", json={"query": "x"}).status_code == 422
    assert client.post("/search", json={"query": "same day", "mode": "sql"}).status_code == 422


def test_find(client):
    body = client.post("/find", json={"description": "man relives the same winter day"}).json()
    assert body["answer"]["candidates"][0]["title"] == "Loop Day"
    assert body["evidence"]["m00000"]["snippet"]


def test_find_stream_emits_events(client):
    with client.stream("POST", "/find/stream", json={"description": "man relives the same winter day"}) as r:
        text = "".join(r.iter_text())
    kinds = [line.split(":", 1)[1].strip() for line in text.splitlines() if line.startswith("event:")]
    assert kinds == ["tool_call", "tool_result", "final"]
