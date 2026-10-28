"""Shared test helpers: a fake OpenAI client that replays scripted replies."""
from __future__ import annotations

from types import SimpleNamespace

import pytest


class FakeClient:
    """Mimics `client.chat.completions.create(...)`, returning scripted replies in order."""

    def __init__(self, replies: list[str]):
        self._replies = list(replies)
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        content = self._replies.pop(0)
        msg = SimpleNamespace(content=content, tool_calls=None)
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
