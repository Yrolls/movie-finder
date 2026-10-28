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


# ---------------------------------------------------------------- multi-query
MULTI_PROMPT = """The user half-remembers a film. Write THREE different short English versions of
their description, each a 1-2 sentence plot summary in the style of a Wikipedia "Plot" section,
so that a search over plot summaries finds the film from different angles:
1. a faithful translation/rewrite of everything they said;
2. the core premise only, in plain general words (who, what situation, what goal);
3. the most specific, distinctive scene or detail they mentioned, described concretely.

Rules:
- Use only what the user said. Do not add details, and do NOT name or guess the movie, its
  title, actors or character names -- even if you think you know it.
Reply with ONLY: {"plots": ["<version 1>", "<version 2>", "<version 3>"]}"""


class MultiRewrite(BaseModel):
    plots: list[str] = Field(min_length=1, max_length=3)


def rewrite_queries(description: str, client=None) -> list[str]:
    out = chat_json(
        [{"role": "system", "content": MULTI_PROMPT}, {"role": "user", "content": description}],
        MultiRewrite,
        client=client,
    )
    return [p.strip() for p in out.plots if len(p.strip()) >= 10]
