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


def test_search_tool_accepts_several_phrasings():
    box = ToolBox(make_retriever())
    out = json.loads(box.run("search_movies", json.dumps(
        {"query": "weatherman relives the same day", "queries": ["small town in winter"]})))
    assert out["results"][0]["movie_id"] == "m00000"
    bad = json.loads(box.run("search_movies", json.dumps({"queries": ["x"]})))
    assert "invalid arguments" in bad["error"]


# ---------------------------------------------------------------- SQL tool + guards
DB_SQL = json.dumps({"sql": "SELECT movie_id, title, year, director, actors FROM movies "
                            "WHERE actors LIKE '%DiCaprio%'"})
QUERY_DB = tool_call("query_movie_database", json.dumps({"question": "movies with Leonardo DiCaprio"}))


def test_query_database_then_restricted_plot_search(fake_client):
    client = fake_client([DB_SQL])                    # consumed by the SQL generator inside the tool
    box = ToolBox(make_retriever(), client=client)
    out = json.loads(box.run("query_movie_database", {"question": "movies with Leonardo DiCaprio"}))
    assert {r["movie_id"] for r in out["results"]} == {"m00001", "m00002"} and out["n_matches"] == 2
    assert box.seen_meta and box.last_progress
    res = json.loads(box.run("search_movies", {"query": "ship strikes an iceberg",
                                               "within_database_results": True}))
    assert {r["movie_id"] for r in res["results"]} <= {"m00001", "m00002"}   # pre-filtered
    assert res["results"][0]["movie_id"] == "m00002"


def test_within_database_needs_a_query_first():
    box = ToolBox(make_retriever())
    out = json.loads(box.run("search_movies", {"query": "ship ice", "within_database_results": True}))
    assert "query_movie_database first" in out["error"]


def test_bad_movie_ids_rejected():
    box = ToolBox(make_retriever())
    out = json.loads(box.run("search_movies", {"query": "ship ice", "movie_ids": ["DROP"]}))
    assert "invalid arguments" in out["error"]


def test_identical_call_is_refused():
    box = ToolBox(make_retriever())
    box.run("search_movies", {"query": "same winter day"})
    out = json.loads(box.run("search_movies", {"query": "  Same Winter DAY "}))
    assert "already made exactly this call" in out["error"] and box.repeats == 1
    assert not box.last_progress


def test_no_progress_stops_the_loop(fake_client):
    client = fake_client([SEARCH, SEARCH, SEARCH, final()])   # 2 repeated searches in a row
    res = run_agent("same day again", make_retriever(), client=client, max_steps=5)
    assert res.stopped_by == "no_progress" and res.steps == 3
    assert "tools" not in client.calls[-1]                    # forced, tool-free final turn
    assert res.answer.status == "found"


def test_metadata_only_answer_is_not_called_found(fake_client):
    # the agent answers from the DB row alone, never checking the story -> downgraded
    client = fake_client([QUERY_DB, DB_SQL, final(movie_id="m00002", chunk="m00002#meta")])
    events = []
    res = run_agent("a DiCaprio film", make_retriever(), client=client, on_event=events.append)
    top = res.answer.candidates[0]
    assert top.movie_id == "m00002" and top.evidence_chunk_id == "m00002#meta"
    assert res.answer.status == "not_sure" and res.answer.follow_up_question
    assert res.evidence["m00002"]["source"] == "metadata"
    assert "sql" in next(e for e in events if e["type"] == "tool_result")


def test_large_sql_result_is_sampled_but_fully_searchable(fake_client, monkeypatch):
    from movie_finder.agent import tools
    monkeypatch.setattr(tools, "DB_SAMPLE", 1)
    box = ToolBox(make_retriever(), client=fake_client([DB_SQL]))
    out = json.loads(box.run("query_movie_database", {"question": "DiCaprio films"}))
    assert out["n_matches"] == 2 and len(out["results"]) == 1 and "showing 1 of 2" in out["note"]
    res = json.loads(box.run("search_movies", {"query": "ship strikes an iceberg",
                                               "within_database_results": True}))
    assert res["results"][0]["movie_id"] == "m00002"      # found although it wasn't in the sample


def test_detail_calls_are_budgeted():
    box = ToolBox(make_retriever())
    for mid in ("m00000", "m00001"):
        assert "error" not in json.loads(box.run("get_movie", {"movie_id": mid}))
    out = json.loads(box.run("get_movie", {"movie_id": "m00002"}))
    assert "different, simpler wording" in out["error"] and not box.last_progress


def test_search_within_also_shows_outside_matches(fake_client):
    # remembered actor (DiCaprio) is "wrong": the story is the weatherman one (m00000, Bill Murray)
    box = ToolBox(make_retriever(), client=fake_client([DB_SQL]))
    box.run("query_movie_database", {"question": "DiCaprio films"})
    out = json.loads(box.run("search_movies", {"query": "weatherman relives the same winter day",
                                               "within_database_results": True}))
    assert {r["movie_id"] for r in out["results"]} <= {"m00001", "m00002"}
    outside = [r["movie_id"] for r in out["outside_database_results"]]
    assert outside[0] == "m00000" and not set(outside) & {"m00001", "m00002"}
    assert "m00000" in box.seen_hits                      # so the agent may answer with it


def test_model_sees_rank_not_similarity(fake_client):
    box = ToolBox(make_retriever())
    out = json.loads(box.run("search_movies", {"query": "same winter day again"}))
    assert [r["rank"] for r in out["results"]] == list(range(1, len(out["results"]) + 1))
    assert all("score" not in r for r in out["results"])          # the model never sees it
    # ...but the UI / trace still get the similarity
    events = []
    run_agent("same day again", make_retriever(), client=fake_client([SEARCH, final()]),
              on_event=events.append)
    hit = next(e for e in events if e["type"] == "tool_result")["hits"][0]
    assert isinstance(hit["score"], float) and hit["rank"] == 1


def test_ui_hits_collapse_duplicate_corpus_rows():
    """The corpus has some films twice under different ids; the UI shows each once."""
    from types import SimpleNamespace
    from movie_finder.agent.loop import _box_hits
    from movie_finder.schemas import SearchHit

    def hit(mid):
        return SearchHit(movie_id=mid, title="Mad Max: Fury Road", year=2015, score=0.84,
                         chunk_id=mid + "#1", snippet="war rig")
    box = SimpleNamespace(last_hits=[], last_outside=[hit("m16583"), hit("m16852")], last_db_sample=[], last_missing=[])
    rows = _box_hits(box)
    assert len(rows) == 1 and rows[0]["outside"] is True


# ---------------------------------------------------------------- check_title
def test_check_title_returns_only_films_in_the_database():
    box = ToolBox(make_retriever())
    out = json.loads(box.run("check_title", json.dumps(
        {"titles": ["the dream heist (2010)", "Totally Made Up Film"], "description": "thieves enter dreams"})))
    found = [r for r in out["results"] if r["in_database"]]
    missing = [r for r in out["results"] if not r["in_database"]]
    assert [r["movie_id"] for r in found] == ["m00001"]           # normalised title + year
    assert found[0]["chunk_id"].startswith("m00001#")            # real evidence passage
    assert [r["title_asked"] for r in missing] == ["Totally Made Up Film"]
    assert box.title_checked == {"m00001"} and box.searched == set()
    assert "score" not in json.dumps(out)


def test_check_title_validates_args():
    box = ToolBox(make_retriever())
    assert "invalid arguments" in box.run("check_title", json.dumps({"titles": [], "description": "x y z"}))
    assert "invalid arguments" in box.run("check_title", json.dumps({"titles": ["a", "b", "c", "d"],
                                                                     "description": "x y z"}))


def test_answer_found_by_title_check_is_labelled(fake_client):
    check = tool_call("check_title", json.dumps({"titles": ["Dream Heist"], "description": "dream thieves"}))
    client = fake_client([check, final(movie_id="m00001", chunk="m00001#0")])
    events = []
    res = run_agent("creepy dream movie", make_retriever(), client=client, on_event=events.append)
    assert res.answer.candidates[0].title == "Dream Heist"
    assert res.evidence["m00001"]["via"] == "title_check"
    assert events[1]["hits"][0]["title"] == "Dream Heist"


def test_title_the_model_invents_cannot_be_the_answer(fake_client):
    check = tool_call("check_title", json.dumps({"titles": ["Annabelle"], "description": "creepy doll"}))
    client = fake_client([check, final(movie_id="m00042", chunk="m00042#0")])
    events = []
    res = run_agent("creepy doll", make_retriever(), client=client, on_event=events.append)
    assert res.answer.status == "not_found" and res.answer.candidates == []
    assert events[1]["hits"] == [{"rank": None, "movie_id": None, "title": "Annabelle", "year": None,
                                  "score": None, "missing": True}]


def test_found_by_search_is_labelled_search(fake_client):
    client = fake_client([SEARCH, final()])
    res = run_agent("same day again", make_retriever(), client=client)
    assert res.evidence["m00000"]["via"] == "search"


def test_second_look_once_before_settling_for_not_sure(fake_client):
    check = tool_call("check_title", json.dumps({"titles": ["Loop Day"], "description": "same day"}))
    client = fake_client([SEARCH, final(status="not_sure", follow_up_question="when?"), check, final()])
    res = run_agent("some movie where a day repeats", make_retriever(), client=client)
    assert res.answer.status == "found" and res.answer.candidates[0].title == "Loop Day"
    assert any("second_look" in t for t in res.trace)
    from movie_finder.agent.loop import SECOND_LOOK
    assert sum(m.get("content") == SECOND_LOOK for m in client.calls[-1]["messages"]) == 1


def test_second_look_is_asked_only_once(fake_client):
    unsure = final(status="not_sure", follow_up_question="when?")
    client = fake_client([SEARCH, unsure, unsure])
    res = run_agent("vague", make_retriever(), client=client)
    assert res.answer.status == "not_sure" and len(client.calls) == 3
    assert sum("second_look" in t for t in res.trace) == 1


def test_no_second_look_after_title_check(fake_client):
    check = tool_call("check_title", json.dumps({"titles": ["Nope"], "description": "x y z"}))
    client = fake_client([check, final(status="not_found")])
    res = run_agent("vague", make_retriever(), client=client)
    assert res.answer.status == "not_found" and len(client.calls) == 2
