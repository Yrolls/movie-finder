"""The agent's tools: an explicit allow-list, argument validation, and untrusted-output handling.

Only names in TOOL_SPECS can run. Every call's arguments are parsed and validated with a
Pydantic model BEFORE touching the retriever; bad calls come back to the model as an error
message instead of raising. The toolbox also records which movies/chunks were actually
returned, so the final answer can be checked against them (no invented citations).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from ..retrieval.retriever import Mode, Retriever
from ..schemas import GetMovieArgs, SearchHit, SearchMoviesArgs

MAX_RESULT_CHARS = 4000          # cap what goes back into the model's context
PLOT_PREVIEW_WORDS = 180

TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_movies",
            "description": (
                "Search a database of movie plot summaries (films from 1980 to 2017) with a "
                "description of what happens in the film. Returns up to k candidate movies, each "
                "with a similarity score and the matching plot passage (chunk_id + snippet)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What happens in the movie, in English, in plain words. "
                                       "Rephrase the user's description; do not guess titles.",
                    },
                    "year_from": {"type": "integer", "description": "Optional earliest release year."},
                    "year_to": {"type": "integer", "description": "Optional latest release year."},
                    "k": {"type": "integer", "description": "Number of results, 1-10 (default 5)."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_movie",
            "description": "Get details for one movie_id returned by search_movies: title, year, "
                           "genre, director, cast and the start of its plot.",
            "parameters": {
                "type": "object",
                "properties": {"movie_id": {"type": "string", "description": "e.g. m01234"}},
                "required": ["movie_id"],
            },
        },
    },
]
ALLOWED_TOOLS = frozenset(t["function"]["name"] for t in TOOL_SPECS)


@dataclass
class ToolBox:
    retriever: Retriever
    mode: Mode = "dense"
    seen_hits: dict[str, SearchHit] = field(default_factory=dict)        # movie_id -> best hit
    seen_chunks: set[str] = field(default_factory=set)
    chunk_text: dict[str, str] = field(default_factory=dict)            # chunk_id -> snippet shown
    chunk_score: dict[str, float] = field(default_factory=dict)         # chunk_id -> retrieval score

    def run(self, name: str, raw_args: str | dict | None) -> str:
        """Execute one tool call and return a JSON string for the model. Never raises."""
        if name not in ALLOWED_TOOLS:
            return _error(f"unknown tool {name!r}; allowed: {sorted(ALLOWED_TOOLS)}")
        try:
            args = raw_args if isinstance(raw_args, dict) else json.loads(raw_args or "{}")
        except json.JSONDecodeError:
            return _error("arguments were not valid JSON")
        try:
            if name == "search_movies":
                return self._search(SearchMoviesArgs.model_validate(args))
            return self._get_movie(GetMovieArgs.model_validate(args))
        except ValidationError as e:
            return _error(f"invalid arguments: {e.errors(include_url=False)}")

    # ------------------------------------------------------------ tools
    def _search(self, a: SearchMoviesArgs) -> str:
        # one phrasing is what the app uses: the eval found single rewrite + hybrid beat
        # 3 fused phrasings (recall@5 75.0% vs 68.8%); `queries` stays accepted for experiments
        hits = self.retriever.search_many(
            a.all_queries, k=a.k, mode=self.mode, year_from=a.year_from, year_to=a.year_to
        )
        for h in hits:
            self.seen_chunks.add(h.chunk_id)
            self.chunk_text[h.chunk_id] = h.snippet
            self.chunk_score[h.chunk_id] = max(h.score, self.chunk_score.get(h.chunk_id, h.score))
            best = self.seen_hits.get(h.movie_id)
            if best is None or h.score > best.score:
                self.seen_hits[h.movie_id] = h
        if not hits:
            return json.dumps({"results": [], "note": "no movie matched closely enough"})
        payload = {"results": [h.model_dump() for h in hits]}
        return _cap(json.dumps(payload, ensure_ascii=False))

    def _get_movie(self, a: GetMovieArgs) -> str:
        m = self.retriever.get_movie(a.movie_id)
        if m is None:
            return _error(f"no movie with id {a.movie_id}")
        plot = " ".join(m.plot.split()[:PLOT_PREVIEW_WORDS])
        payload = {
            "movie_id": m.movie_id, "title": m.title, "year": m.year, "genre": m.genre,
            "director": m.director, "cast": m.cast[:200], "plot_start": plot,
        }
        return _cap(json.dumps(payload, ensure_ascii=False))


def _error(msg: str) -> str:
    return json.dumps({"error": msg})


def _cap(s: str) -> str:
    return s if len(s) <= MAX_RESULT_CHARS else s[:MAX_RESULT_CHARS] + " ...[truncated]"
