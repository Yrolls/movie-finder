"""The agent's tools: an explicit allow-list, argument validation, and untrusted-output handling.

Tools: search_movies (plot search), query_movie_database (guarded text-to-SQL over metadata),
check_title (verify films the model thinks of, from its own knowledge), get_movie (details). Only names in the toolbox's specs can run. Every call's arguments are parsed and validated with a
Pydantic model BEFORE touching the retriever; bad calls come back to the model as an error
message instead of raising. The toolbox also records which movies/chunks were actually
returned, so the final answer can be checked against them (no invented citations).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from ..config import get_settings
from ..data.moviedb import MovieDB
from ..retrieval.retriever import Mode, Retriever
from ..schemas import CheckTitleArgs, GetMovieArgs, QueryDatabaseArgs, SearchHit, SearchMoviesArgs
from ..titles import TitleIndex, split_year
from .sql_tool import answer_with_sql

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
                "with its rank and the matching plot passage (chunk_id + snippet)."
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
                    "within_database_results": {
                        "type": "boolean",
                        "description": "Set true to search only the plots of the films your last "
                                       "query_movie_database call matched (all of them, not just "
                                       "the ones shown).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_movie_database",
            "description": (
                "Find movies by facts the user remembers that are NOT about the plot: an actor, the "
                "director, the year or decade, the country, the genre. Pass a short English question, "
                "e.g. 'movies with Tom Hanks from the 90s'. Returns how many films match and a "
                "sample of them. Then call search_movies with within_database_results=true to see "
                "which of those films has a plot that fits the user's description."
            ),
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string",
                                            "description": "The non-plot facts, in English."}},
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_title",
            "description": (
                "Check films YOU think it might be, from your own knowledge of movies. Use it when "
                "the description is mostly about how something looked or felt (a creepy doll, a "
                "costume, a famous image) or is too vague for a plot search, and you can think of "
                "likely films. Returns, for each title, whether it is in the database and, if so, "
                "the plot passage closest to the description, so you can confirm or reject it. "
                "A film that is not in the database can never be the answer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "titles": {"type": "array", "items": {"type": "string"},
                               "description": "1-3 English film titles, optionally with the year, "
                                              "e.g. [\"Annabelle (2014)\", \"Dead Silence\"]"},
                    "description": {"type": "string",
                                    "description": "The user's description in English."},
                },
                "required": ["titles", "description"],
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
META = "#meta"                    # chunk-id suffix for a citation that is a metadata row
DB_SAMPLE = 15                    # SQL rows shown to the model (all matches are kept in db_ids)
MAX_DETAIL_CALLS = 2              # get_movie budget per run: re-search instead of browsing
OUTSIDE_K = 3                     # best plot matches outside the SQL set, shown next to it


def get_title_index(retriever: Retriever) -> TitleIndex:
    idx = getattr(retriever, "_title_index", None)
    if idx is None:
        idx = TitleIndex(retriever.movies.values())
        retriever._title_index = idx
    return idx


def get_moviedb(retriever: Retriever) -> MovieDB:
    """One read-only metadata DB per retriever, built on first use (~0.2 s for 18k movies)."""
    db = getattr(retriever, "_moviedb", None)
    if db is None:
        db = MovieDB(retriever.movies.values())
        retriever._moviedb = db
    return db


@dataclass
class ToolBox:
    retriever: Retriever
    mode: Mode = "dense"
    client: Any = None                                                  # LLM client for the SQL tool
    seen_hits: dict[str, SearchHit] = field(default_factory=dict)        # movie_id -> best plot hit
    seen_meta: dict[str, str] = field(default_factory=dict)             # movie_id -> metadata line
    seen_chunks: set[str] = field(default_factory=set)
    chunk_text: dict[str, str] = field(default_factory=dict)            # chunk_id -> snippet shown
    chunk_score: dict[str, float] = field(default_factory=dict)         # chunk_id -> retrieval score
    # no-progress guard (Lectures 10-11): identical calls are refused, and the loop watches
    # `last_progress` to stop an agent that keeps calling tools without learning anything new
    db_ids: list[str] | None = None                                      # last SQL match set
    detail_calls: int = 0
    last_hits: list[SearchHit] = field(default_factory=list)            # for the UI / trace only
    last_outside: list[SearchHit] = field(default_factory=list)
    last_db_sample: list[str] = field(default_factory=list)
    last_missing: list[str] = field(default_factory=list)                # titles check_title did not find
    searched: set[str] = field(default_factory=set)                     # ids a plot/DB search returned
    title_checked: set[str] = field(default_factory=set)                # ids found by check_title
    calls_seen: set[str] = field(default_factory=set)
    repeats: int = 0
    last_progress: bool = False

    def specs(self) -> list[dict[str, Any]]:
        return TOOL_SPECS

    def run(self, name: str, raw_args: str | dict | None) -> str:
        """Execute one tool call and return a JSON string for the model. Never raises."""
        self.last_progress = False
        self.last_hits, self.last_outside, self.last_db_sample, self.last_missing = [], [], [], []
        if name not in ALLOWED_TOOLS:
            return _error(f"unknown tool {name!r}; allowed: {sorted(ALLOWED_TOOLS)}")
        try:
            args = raw_args if isinstance(raw_args, dict) else json.loads(raw_args or "{}")
        except json.JSONDecodeError:
            return _error("arguments were not valid JSON")
        if not isinstance(args, dict):
            return _error("arguments must be a JSON object")

        sig = _signature(name, args)
        if sig in self.calls_seen:
            self.repeats += 1
            return _error("you already made exactly this call and saw its result; use different "
                          "wording, or stop and give your final JSON answer")
        self.calls_seen.add(sig)

        before = len(self.seen_hits.keys() | self.seen_meta.keys())
        try:
            if name == "search_movies":
                within = bool(args.pop("within_database_results", False))
                args.setdefault("k", get_settings().top_k)           # TOP_K unless the model asks
                a = SearchMoviesArgs.model_validate(args)
                if within:
                    if not self.db_ids:
                        return _error("no database results to search within; call "
                                      "query_movie_database first, or search without it")
                    out = self._search_within(a)
                else:
                    out = self._search(a)
            elif name == "query_movie_database":
                out = self._query_db(QueryDatabaseArgs.model_validate(args))
            elif name == "check_title":
                out = self._check_title(CheckTitleArgs.model_validate(args))
            else:
                if self.detail_calls >= MAX_DETAIL_CALLS:
                    return _error(f"you already checked {MAX_DETAIL_CALLS} movies in detail. If one "
                                  "fits, answer now; if none fits, call search_movies with "
                                  "different, simpler wording")
                self.detail_calls += 1
                out = self._get_movie(GetMovieArgs.model_validate(args))
                self.last_progress = '"error"' not in out[:12]   # inspecting a new movie counts
                return out
        except ValidationError as e:
            return _error(f"invalid arguments: {e.errors(include_url=False)}")
        self.last_progress = len(self.seen_hits.keys() | self.seen_meta.keys()) > before
        return out

    # ------------------------------------------------------------ tools
    def _search(self, a: SearchMoviesArgs) -> str:
        hits = self._retrieve(a)
        self.last_hits = hits
        if not hits:
            return json.dumps({"results": [], "note": "no movie matched closely enough"})
        payload = {"results": _for_model(hits),
                   "note": "results are ordered best match first (rank 1 = best)"}
        return _cap(json.dumps(payload, ensure_ascii=False))

    def _search_within(self, a: SearchMoviesArgs) -> str:
        """Plot search inside the SQL match set, plus the best plot matches OUTSIDE it.

        People misremember actors and directors (eval q48: "the Mad Max with Mel Gibson" is
        Fury Road, with Tom Hardy). Showing both lets the model notice when the story fits a
        film outside the remembered facts much better.
        """
        inside = self._retrieve(a.model_copy(update={"movie_ids": self.db_ids}))
        ids = set(self.db_ids or [])
        outside = [h for h in self._retrieve(a.model_copy(update={"k": OUTSIDE_K + len(inside)}))
                   if h.movie_id not in ids][:OUTSIDE_K]
        self.last_hits, self.last_outside = inside, outside
        payload = {
            "results": _for_model(inside),
            "outside_database_results": _for_model(outside),
            "note": ("results = films matching the remembered facts; outside_database_results = "
                     "best plot matches without them. If an outside film fits the story clearly "
                     "better, the remembered actor/director/year may be wrong."),
        }
        return _cap(json.dumps(payload, ensure_ascii=False))

    def _retrieve(self, a: SearchMoviesArgs) -> list[SearchHit]:
        # one phrasing is what the app uses: the eval found single rewrite + hybrid beat
        # 3 fused phrasings (recall@5 75.0% vs 68.8%); `queries` stays accepted for experiments
        hits = self.retriever.search_many(
            a.all_queries, k=a.k, mode=self.mode, year_from=a.year_from, year_to=a.year_to,
            movie_ids=a.movie_ids,
        )
        self.searched.update(h.movie_id for h in hits)
        self._record(hits)
        return hits

    def _record(self, hits: list[SearchHit]) -> None:
        for h in hits:
            self.seen_chunks.add(h.chunk_id)
            self.chunk_text[h.chunk_id] = h.snippet
            self.chunk_score[h.chunk_id] = max(h.score, self.chunk_score.get(h.chunk_id, h.score))
            best = self.seen_hits.get(h.movie_id)
            if best is None or h.score > best.score:
                self.seen_hits[h.movie_id] = h

    def _query_db(self, a: QueryDatabaseArgs) -> str:
        res = answer_with_sql(a.question, get_moviedb(self.retriever), client=self.client)
        if "error" in res:
            return json.dumps({"sql": res["sql"], "error": res["error"],
                               "note": "the metadata query failed; continue with search_movies"})
        ids = []
        for row in res["rows"]:
            mid = dict(zip(res["columns"], row)).get("movie_id")
            if mid and self.retriever.get_movie(mid) is not None and mid not in ids:
                ids.append(mid)                  # only real corpus movies, in query order
        self.db_ids = ids or None
        sample = []
        for mid in ids[:DB_SAMPLE]:
            m = self.retriever.get_movie(mid)
            line = f"{m.title} ({m.year}) — directed by {m.director or 'unknown'}; cast: {m.cast[:150] or 'unknown'}"
            self.seen_meta[mid] = line
            self.seen_chunks.add(mid + META)
            self.chunk_text[mid + META] = line
            sample.append({"movie_id": mid, "title": m.title, "year": m.year})
        self.last_db_sample = [r["movie_id"] for r in sample]
        payload = {"sql": res["sql"], "n_matches": len(ids), "results": sample}
        if res.get("relaxed_from"):
            payload["relaxed_from"] = res["relaxed_from"]       # first query matched nothing
        if not ids:
            payload["note"] = ("no film in the database matches these facts -- the user may "
                               "misremember them; continue with a plain plot search")
        elif len(ids) > DB_SAMPLE:
            payload["note"] = (f"showing {DB_SAMPLE} of {len(ids)}; call search_movies with "
                               "within_database_results=true to search all of them by plot")
        return _cap(json.dumps(payload, ensure_ascii=False))

    def _check_title(self, a: CheckTitleArgs) -> str:
        """Verify the model's own guesses against the corpus; cite the passage closest to the description.

        Plot summaries rarely describe how things look, so "a creepy doll with blonde hair" finds
        nothing by plot search even when the model knows the film. Here the model names films, and
        only films that exist in the database come back -- with real evidence, never invented.
        """
        index = get_title_index(self.retriever)
        found, results = [], []
        for raw in dict.fromkeys(t.strip() for t in a.titles):
            title, year = split_year(raw)
            # the whole string first: "Blade Runner 2049" is a title, not "Blade Runner" + a year
            movies = index.find(raw) or (index.find(title, year) if year else [])
            if not movies:
                self.last_missing.append(raw)
                results.append({"title_asked": raw, "in_database": False})
                continue
            ids = [m.movie_id for m in movies[:2]]
            hits = self.retriever.search(a.description, k=len(ids), mode="dense", movie_ids=ids,
                                         min_score=0.0, rerank=False)
            for h in hits:
                found.append(h)
                results.append({"title_asked": raw, "in_database": True, "movie_id": h.movie_id,
                                "title": h.title, "year": h.year, "chunk_id": h.chunk_id,
                                "snippet": h.snippet})
        self._record(found)
        self.title_checked.update(h.movie_id for h in found)
        self.last_hits = found
        payload = {"results": results,
                   "note": "in_database=false means it cannot be the answer. For films in the "
                           "database, check the passage against the description before answering."}
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


def _for_model(hits: list[SearchHit]) -> list[dict[str, Any]]:
    """What the model sees of a ranked hit list: the RANK, not the raw similarity.

    The list is ordered by hybrid (BM25 + dense, RRF) rank, but the dense similarity of each hit
    can disagree with that order. When the score was shown, the model picked the highest score
    over the top-ranked film (eval: q01 Groundhog Day ranked 1st with 0.814, Whitewash ranked
    3rd with 0.816 -> the agent answered Whitewash). The UI still shows the similarity.
    """
    return [{"rank": i, "movie_id": h.movie_id, "title": h.title, "year": h.year,
             "chunk_id": h.chunk_id, "snippet": h.snippet} for i, h in enumerate(hits, 1)]


def _signature(name: str, args: dict) -> str:
    """Normalised call identity: same tool + same args (case/space-insensitive) = a repeat."""
    def norm(v):
        if isinstance(v, str):
            return " ".join(v.lower().split())
        if isinstance(v, list):
            return sorted(norm(x) for x in v)
        return v
    return name + json.dumps({k: norm(v) for k, v in args.items()}, sort_keys=True, default=str)


def _error(msg: str) -> str:
    return json.dumps({"error": msg})


def _cap(s: str) -> str:
    return s if len(s) <= MAX_RESULT_CHARS else s[:MAX_RESULT_CHARS] + " ...[truncated]"
