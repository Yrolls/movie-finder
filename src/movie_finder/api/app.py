"""FastAPI app: REST endpoints + an SSE stream of the agent's steps + a one-page demo UI.

    uvicorn movie_finder.api.app:app --app-dir src --port 8000
    open http://localhost:8000

Endpoints
  GET  /health              index + model info
  POST /search              retrieval only (no LLM)          -> list[SearchHit]
  POST /find                full agent, one JSON response    -> FinalAnswer + steps
  POST /find/stream         full agent as Server-Sent Events: tool_call, tool_result, final
"""
from __future__ import annotations

import asyncio
import json
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from ..agent.loop import run_agent
from ..config import get_settings
from ..llm import LLMOutputError
from ..retrieval.retriever import Retriever
from ..schemas import FinalAnswer, FindRequest, SearchHit

STATIC = Path(__file__).parent / "static"


class SearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=500)
    k: int | None = Field(default=None, ge=1, le=10)                   # default: TOP_K
    mode: Literal["dense", "bm25", "hybrid"] | None = None             # default: RETRIEVAL_MODE
    year_from: int | None = Field(default=None, ge=1890, le=2030)
    year_to: int | None = Field(default=None, ge=1890, le=2030)


class FindResponse(BaseModel):
    answer: FinalAnswer
    evidence: dict[str, dict[str, Any]]       # movie_id -> chunk_id, passage, snippet, similarity, wiki_url
    steps: int
    stopped_by: str


def create_app(retriever: Retriever | None = None) -> FastAPI:
    """`retriever` can be injected (tests); otherwise it is loaded from data/ at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if retriever is not None:
            app.state.retriever = retriever
        else:
            from ..retrieval.loader import load_retriever

            app.state.retriever = await asyncio.to_thread(load_retriever)
        yield

    app = FastAPI(title="Movie Finder", lifespan=lifespan)

    def _r(request: Request) -> Retriever:
        return request.app.state.retriever

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/health")
    def health(request: Request):
        r = _r(request)
        s = get_settings()
        return {"status": "ok", "movies": len(r.movies), "chunks": len(r.store.chunks),
                "index": r.store.meta, "llm_model": s.llm_model, "min_score": r.min_score}

    @app.post("/search", response_model=list[SearchHit])
    def search(body: SearchRequest, request: Request):
        s = get_settings()
        return _r(request).search(body.query, k=body.k or s.top_k, mode=body.mode or s.retrieval_mode,
                                  year_from=body.year_from, year_to=body.year_to)

    @app.post("/find", response_model=FindResponse)
    def find(body: FindRequest, request: Request):
        try:
            res = run_agent(body.description, _r(request), history=body.history)
        except LLMOutputError as e:
            raise HTTPException(status_code=502, detail=f"model output error: {e}") from e
        return FindResponse(answer=res.answer, evidence=res.evidence, steps=res.steps,
                            stopped_by=res.stopped_by)

    @app.post("/find/stream")
    async def find_stream(body: FindRequest, request: Request):
        """Run the (blocking) agent in a worker thread and forward its events as SSE."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        done = object()

        def emit(event: dict) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, event)

        def worker() -> None:
            try:
                run_agent(body.description, _r(request), history=body.history, on_event=emit)
            except Exception as e:  # noqa: BLE001 -- report any failure to the client, don't hang
                emit({"type": "error", "message": f"{type(e).__name__}: {e}"[:300]})
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, done)

        threading.Thread(target=worker, daemon=True).start()

        async def events():
            while True:
                event = await queue.get()
                if event is done:
                    break
                yield {"event": event["type"], "data": json.dumps(event, ensure_ascii=False)}

        return EventSourceResponse(events())

    return app


app = create_app()
