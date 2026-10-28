import pytest
from pydantic import ValidationError

from movie_finder import config
from movie_finder.llm import LLMOutputError, chat_json, extract_json
from movie_finder.schemas import FinalAnswer, SearchMoviesArgs


# ---------------------------------------------------------------- config
def test_settings_defaults():
    s = config.get_settings()
    assert s.llm_model == "gemma-4-E4B-it"
    assert s.top_k == 5


def test_bad_overlap_rejected(monkeypatch):
    monkeypatch.setenv("CHUNK_WORDS", "100")
    monkeypatch.setenv("CHUNK_OVERLAP", "100")
    with pytest.raises(config.ConfigError):
        config.get_settings()


def test_missing_key_is_clear(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    with pytest.raises(config.ConfigError, match="OPENAI_API_KEY"):
        config.get_settings().require_llm()


# ---------------------------------------------------------------- JSON extraction
@pytest.mark.parametrize(
    "reply",
    [
        '{"a": 1}',
        '```json\n{"a": 1}\n```',
        'Sure! Here it is:\n{"a": 1}\nHope that helps.',
    ],
)
def test_extract_json_variants(reply):
    assert extract_json(reply) == {"a": 1}


# ---------------------------------------------------------------- schemas
def test_found_requires_candidate():
    with pytest.raises(ValidationError):
        FinalAnswer(status="found", candidates=[])


def test_not_sure_requires_question():
    with pytest.raises(ValidationError):
        FinalAnswer(status="not_sure")


def test_tool_args_reject_bad_year_range():
    with pytest.raises(ValidationError):
        SearchMoviesArgs(query="a man relives the same day", year_from=2000, year_to=1990)


# ---------------------------------------------------------------- chat_json repair loop
GOOD = '{"status": "not_found", "candidates": []}'


def test_chat_json_repairs_once(fake_client):
    client = fake_client(["not json at all", GOOD])
    ans = chat_json([{"role": "user", "content": "x"}], FinalAnswer, client=client)
    assert ans.status == "not_found"
    assert len(client.calls) == 2
    assert all(c["temperature"] == 0.0 for c in client.calls)


def test_chat_json_gives_up(fake_client):
    client = fake_client(["nope"] * 3)
    with pytest.raises(LLMOutputError):
        chat_json([{"role": "user", "content": "x"}], FinalAnswer, retries=2, client=client)
