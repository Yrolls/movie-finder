"""Shared test helpers: a fake OpenAI client that replays scripted replies."""
from __future__ import annotations

from types import SimpleNamespace

import pytest


def tool_call(name: str, arguments: str, call_id: str = "call_1"):
    """A scripted reply that asks to run one tool."""
    fn = SimpleNamespace(name=name, arguments=arguments)
    return {"tool_calls": [SimpleNamespace(id=call_id, type="function", function=fn)]}


class FakeClient:
    """Mimics `client.chat.completions.create(...)`, returning scripted replies in order.

    A reply is either a string (plain content) or a dict from `tool_call(...)`.
    """

    def __init__(self, replies: list):
        self._replies = list(replies)
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        r = self._replies.pop(0)
        if isinstance(r, dict):
            msg = SimpleNamespace(content="", tool_calls=r["tool_calls"])
        else:
            msg = SimpleNamespace(content=r, tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


@pytest.fixture
def fake_client():
    return FakeClient


@pytest.fixture(autouse=True)
def _clean_settings(monkeypatch):
    """Each test gets fresh settings and never reads the developer's real .env key."""
    from movie_finder import config

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost/v1")
    config.get_settings.cache_clear()
    yield
    config.get_settings.cache_clear()
