import json

from movie_finder.agent.loop import run_agent
from movie_finder.agent.tools import ToolBox
from tests.conftest import tool_call
from tests.fixtures import make_retriever

SEARCH = tool_call("search_movies", json.dumps({"query": "man relives the same winter day"}))


def final(status="found", movie_id="m00000", chunk="m00000#0", title="WRONG TITLE", **extra):
    cands = [] if status == "not_found" else [
        {"movie_id": movie_id, "title": title, "year": 1900, "confidence": 0.9,
         "evidence_chunk_id": chunk, "why": "same day repeats"}]
    return json.dumps({"status": status, "candidates": cands, **extra})


# ---------------------------------------------------------------- tools
def test_toolbox_rejects_unknown_tool():
    box = ToolBox(make_retriever())
    assert "unknown tool" in box.run("delete_everything", "{}")


def test_toolbox_validates_args_before_running():
    box = ToolBox(make_retriever())
    out = json.loads(box.run("search_movies", json.dumps({"query": "x", "k": 99})))
    assert "invalid arguments" in out["error"]
    assert box.seen_hits == {}                                 # retriever never called
    assert "not valid JSON" in box.run("search_movies", "{oops")


def test_get_movie_bad_id():
    box = ToolBox(make_retriever())
    assert "invalid arguments" in box.run("get_movie", json.dumps({"movie_id": "../etc"}))
    assert "no movie" in box.run("get_movie", json.dumps({"movie_id": "m99999"}))


# ---------------------------------------------------------------- loop
def test_search_then_answer(fake_client):
    client = fake_client([SEARCH, final()])
    events = []
    res = run_agent("ผู้ชายตื่นมาเจอวันเดิมซ้ำ ๆ", make_retriever(), client=client, on_event=events.append)
    assert res.stopped_by == "final_answer" and res.steps == 2
    top = res.answer.candidates[0]
    assert top.title == "Loop Day" and top.year == 1993          # corpus values, not the model's
    assert client.calls[0]["tools"]                              # tools offered to the model
    assert client.calls[1]["messages"][-1]["role"] == "tool"     # tool result fed back
    assert [e["type"] for e in events] == ["tool_call", "tool_result", "final"]


def test_invented_movie_is_dropped(fake_client):
    search_one = tool_call("search_movies", json.dumps({"query": "man relives the same winter day", "k": 1}))
    client = fake_client([search_one, final(movie_id="m00002", chunk="m00002#0")])  # never retrieved
    res = run_agent("same day again", make_retriever(), client=client)
    assert res.answer.status == "not_found"
    assert res.answer.candidates == []


def test_bad_citation_replaced_with_real_chunk(fake_client):
    client = fake_client([SEARCH, final(chunk="m00001#0")])     # chunk from another movie
    res = run_agent("same day again", make_retriever(), client=client)
    assert res.answer.candidates[0].evidence_chunk_id.startswith("m00000#")


def test_invalid_final_is_repaired(fake_client):
    client = fake_client([SEARCH, "I think it's Loop Day!", final()])
    res = run_agent("same day again", make_retriever(), client=client)
    assert res.answer.status == "found" and res.steps == 3


def test_max_steps_stops_the_loop(fake_client):
    client = fake_client([SEARCH, SEARCH, final()])             # 2 tool turns, then forced final
    res = run_agent("same day again", make_retriever(), client=client, max_steps=2)
    assert res.stopped_by == "max_steps"
    assert "tools" not in client.calls[-1]                      # forced turn has no tools
    assert res.answer.candidates[0].movie_id == "m00000"


def test_garbage_after_max_steps_falls_back_to_hits(fake_client):
    client = fake_client([SEARCH, "nope", "still nope"])
    res = run_agent("same day again", make_retriever(), client=client, max_steps=1)
    assert res.answer.status == "not_sure"
    assert res.answer.candidates[0].movie_id == "m00000"


def test_final_carries_evidence_text(fake_client):
    client = fake_client([SEARCH, final()])
    res = run_agent("same day again", make_retriever(), client=client)
    cited = res.answer.candidates[0].evidence_chunk_id
    chunk_text = {c.chunk_id: c.text for c in make_retriever().store.chunks}[cited]
    assert res.evidence["m00000"]["snippet"] == chunk_text[:300]   # evidence = the cited chunk's text
