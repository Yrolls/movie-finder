"""LLM query rewriting: turn a half-remembered description (any language) into the kind of text
the index contains -- a short English plot summary.

Why: the eval showed raw Thai queries score 0% with multilingual-e5-small (cross-lingual matching
fails) and vague English queries match plots poorly. The agent already rewrites before it calls
search_movies; this module makes the same step measurable on its own.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from ..llm import chat_json

REWRITE_PROMPT = """Rewrite the user's half-remembered description of a film as a short plot
summary in English, in the style of a Wikipedia "Plot" section (2-3 sentences, third person,
plain words). Translate if the user did not write in English.

Rules:
- Keep every concrete detail the user gave (setting, characters, objects, events).
- Do NOT add details the user did not give, and do NOT name or guess the movie, its title,
  actors or character names -- even if you think you know it.
Reply with ONLY: {"plot": "<the rewritten plot>"}"""


class Rewrite(BaseModel):
    plot: str = Field(min_length=10, max_length=1200)


def rewrite_query(description: str, client=None) -> str:
    out = chat_json(
        [{"role": "system", "content": REWRITE_PROMPT}, {"role": "user", "content": description}],
        Rewrite,
        client=client,
    )
    return out.plot
