"""Pydantic models for every boundary in the app.

- corpus records       (Movie, Chunk)
- retrieval results    (SearchHit)
- tool arguments       (SearchMoviesArgs, GetMovieArgs)  -> validated BEFORE a tool runs
- the agent's answer   (FinalAnswer)                     -> validated AFTER the LLM replies
- the HTTP API         (FindRequest)
"""
from __future__ import annotations

import re

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------- corpus
class Movie(BaseModel):
    movie_id: str                      # stable id, e.g. "m01234"
    title: str
    year: int
    origin: str = ""                   # e.g. "American", "British", "Bollywood"
    genre: str = ""
    director: str = ""
    cast: str = ""
    wiki_url: str = ""
    plot: str


class Chunk(BaseModel):
    chunk_id: str                      # "<movie_id>#<n>"
    movie_id: str
    text: str


# ---------------------------------------------------------------- retrieval
class SearchHit(BaseModel):
    movie_id: str
    title: str
    year: int
    score: float
    chunk_id: str                      # the best-matching passage -> used as the citation
    snippet: str


# ---------------------------------------------------------------- tool args
class SearchMoviesArgs(BaseModel):
    query: str | None = Field(default=None, min_length=3, max_length=500)
    queries: list[str] | None = Field(default=None, min_length=1, max_length=3)   # alt phrasings
    year_from: int | None = Field(default=None, ge=1890, le=2030)
    year_to: int | None = Field(default=None, ge=1890, le=2030)
    k: int = Field(default=5, ge=1, le=10)
    movie_ids: list[str] | None = Field(default=None, min_length=1, max_length=2000)  # pre-filter

    @model_validator(mode="after")
    def _year_range(self) -> "SearchMoviesArgs":
        if self.year_from and self.year_to and self.year_from > self.year_to:
            raise ValueError("year_from must be <= year_to")
        if not self.query and not self.queries:
            raise ValueError("give `query` or `queries`")
        if self.queries and any(not 3 <= len(q) <= 500 for q in self.queries):
            raise ValueError("each query must be 3-500 characters")
        if self.movie_ids and any(not re.fullmatch(r"m\d{5}", m) for m in self.movie_ids):
            raise ValueError("movie_ids must look like m01234")
        return self

    @property
    def all_queries(self) -> list[str]:
        return ([self.query] if self.query else []) + list(self.queries or [])


class QueryDatabaseArgs(BaseModel):
    question: str = Field(min_length=5, max_length=300)


class GetMovieArgs(BaseModel):
    movie_id: str = Field(pattern=r"^m\d{5}$")


# ---------------------------------------------------------------- agent output
class Candidate(BaseModel):
    movie_id: str
    title: str
    year: int
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_chunk_id: str             # must be a chunk the agent actually retrieved
    why: str = Field(max_length=400)   # one or two sentences linking the description to the plot


class FinalAnswer(BaseModel):
    status: Literal["found", "not_sure", "not_found"]
    candidates: list[Candidate] = Field(default_factory=list, max_length=3)
    follow_up_question: str | None = None

    @field_validator("candidates")
    @classmethod
    def _sorted(cls, v: list[Candidate]) -> list[Candidate]:
        return sorted(v, key=lambda c: c.confidence, reverse=True)

    @model_validator(mode="after")
    def _consistent(self) -> "FinalAnswer":
        if self.status == "found" and not self.candidates:
            raise ValueError("status 'found' needs at least one candidate")
        if self.status == "not_sure" and not self.follow_up_question:
            raise ValueError("status 'not_sure' needs a follow_up_question")
        return self


# ---------------------------------------------------------------- API
class FindRequest(BaseModel):
    description: str = Field(min_length=5, max_length=2000)
    history: list[str] = Field(default_factory=list, max_length=5)  # earlier answers to follow-ups
