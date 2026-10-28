# 🎬 Movie Finder — "What was that movie called?"

Describe a film you half-remember — in English or Thai — and the app tells you which
movie it is, with the plot passage that matched as evidence. If it isn't confident,
it asks you one follow-up question (e.g. "roughly what decade?") instead of guessing.

> LLM course capstone. Corpus: [Wikipedia Movie Plots](https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots)
> (CC BY-SA 4.0). Model: `gemma-4-E4B-it` via the class OpenAI-compatible proxy.

## Capability families used

| # | Family | Where |
|---|--------|-------|
| 1 | Structured output + schema validation | `src/movie_finder/schemas.py`, `src/movie_finder/llm.py` |
| 2 | Retrieval / RAG with citations | `src/movie_finder/retrieval/` |
| 3 | Retrieval evaluation (recall@k, MRR vs. BM25 baseline) | `eval/` |
| 4 | Agent loop with allow-listed tools + stop condition | `src/movie_finder/agent/` |
| 6 | Served interface (FastAPI + SSE) | `src/movie_finder/api/` |

## Project layout

```
movie-finder/
├── src/movie_finder/
│   ├── config.py        # all settings read from the environment (.env)
│   ├── llm.py           # the ONLY place that talks to the LLM proxy
│   ├── schemas.py       # Pydantic models for every boundary (LLM output, tool args, API)
│   ├── data/            # load + clean + chunk the corpus
│   ├── retrieval/       # embedder, vector store (min_score floor), BM25 baseline
│   ├── agent/           # allow-listed tools + agent loop (max-steps stop condition)
│   └── api/             # FastAPI app (REST + SSE streaming)
├── scripts/             # fetch_data.py, build_index.py, smoke_test_llm.py
├── eval/                # labelled queries, eval runner, committed results
├── tests/               # unit tests (no network / no API key needed)
└── data/                # raw / processed / index — git-ignored, rebuilt by scripts
```

Layering rule: `api → agent → retrieval / llm → config`. Lower layers never import upper ones.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then put your OPENAI_API_KEY / OPENAI_BASE_URL in .env
```

## Run

```bash
# 1. check the class proxy works (JSON output + a Thai prompt)
python scripts/smoke_test_llm.py

# 2. download + clean the corpus  → data/processed/movies.jsonl
python scripts/fetch_data.py

# 3. chunk + embed the plots       → data/index/   (first run downloads the embedding model)
python scripts/build_index.py
```

```bash
# 4. start the app  → open http://localhost:8000
uvicorn movie_finder.api.app:app --app-dir src --port 8000
```

| Endpoint | What it does |
|---|---|
| `GET /` | demo page: type a description, watch the agent's steps stream in, see the answer + cited passage |
| `GET /health` | corpus / index / model info |
| `POST /search` | retrieval only, no LLM — `{"query": "...", "k": 5, "mode": "dense"\|"bm25"\|"hybrid"}` |
| `POST /find` | full agent, one JSON reply — `{"description": "...", "history": []}` |
| `POST /find/stream` | same as `/find`, as Server-Sent Events: `tool_call`, `tool_result`, `final` |

## Evaluate

The gold set is [`eval/queries.jsonl`](eval/queries.jsonl): half-remembered descriptions
(English and Thai) labelled with the movie's title + year, plus queries for movies that are
**not** in the corpus (`"title": null`), where the right answer is "not found".

```bash
python scripts/find_movie.py "groundhog"     # look up a title when adding a query
python scripts/check_dense.py                # sanity check: does a plot passage find its own movie?
python eval/run_eval.py                      # retrieval: BM25 baseline vs dense vs hybrid + min_score sweep
python eval/run_eval.py --rewrite            # + the same queries after LLM rewriting (cached, calls the LLM)
python eval/run_eval.py --rewrite --multi    # + 3 phrasings per query fused with RRF (cached, calls the LLM)
python eval/run_eval.py --rewrite --multi --agent   # + end-to-end agent vs LLM-only baseline

# chunk-size experiment: build extra indexes, then compare them in one report
python scripts/build_index.py --chunk-words 100 --chunk-overlap 25 --out data/index_c100
python scripts/build_index.py --chunk-words 400 --chunk-overlap 100 --out data/index_c400
python eval/run_eval.py --index data/index data/index_c100 data/index_c400
```

Results are written to [`eval/results/report.md`](eval/results/report.md) (committed) and
`eval/results/results.json`.

## Tests

```bash
pytest
```

Unit tests use small fixtures and a fake LLM, so they run without network or an API key.
