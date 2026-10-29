# 🎬 Movie Finder — "What was that movie called?"

Describe a film you half-remember — in **Thai or English** — and the app finds it in a database
of 18,432 film plots (Wikipedia, 1980–2017). Every answer cites the plot passage it matched.
When it is not sure it says so and asks one follow-up question; when the film is not in the
database it says that instead of guessing.

> LLM course capstone. Model: `gemma-4-E4B-it` through the class OpenAI-compatible proxy.
> Corpus: [Wikipedia Movie Plots](https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots) (CC BY-SA 4.0).

**Results** (top-1 accuracy on queries whose film is in the corpus; details in
[`docs/experiments.md`](docs/experiments.md)):

| test set | agent (this app) | LLM-only baseline |
|---|---|---|
| main set, 42 queries (used for tuning) | **88.1%** | 71.4% |
| held-out 1, 16 queries (written by a separate agent, run once) | **68.8%** | 50.0% |
| held-out 2, 14 queries (mostly "what it looked like" memories, run once) | **71.4%** | 35.7% |

Retrieval baseline: BM25 recall@5 42.9% → the app's hybrid + LLM-rewrite search 76.2%.

## How it works

```
your description (TH/EN)
   │
   ▼
agent loop (gemma-4-E4B-it, native tool calling, ≤5 tool turns)
   ├─ search_movies         plot search: LLM-rewritten English query → BM25 + dense (e5) fused with RRF
   │                        → top-5 films, each with its best-matching passage
   ├─ query_movie_database  actor / director / decade / country → guarded text-to-SQL (read-only SQLite)
   │                        → then search_movies only inside those films (+ best matches outside them)
   ├─ check_title           the model names films it knows → the corpus confirms they exist
   │                        → the passage closest to your description becomes the evidence
   └─ get_movie             details of one candidate
   │
   ▼
grounding check: only films a tool returned, only real passages as citations,
title/year taken from the corpus → "found" / "not sure" + follow-up / "not in the database"
```

| capability family | where |
|---|---|
| Structured output + schema validation (JSON repair loop) | `src/movie_finder/llm.py`, `src/movie_finder/schemas.py` |
| RAG with citations: hybrid retrieval, query rewriting, pre-filters, `min_score` floor | `src/movie_finder/retrieval/` |
| Evaluation: recall@k, MRR, nDCG, bootstrap CIs, agent vs. LLM-only, held-out sets | `eval/`, `src/movie_finder/evaluation.py` |
| Agent: allow-listed validated tools, stop conditions (final answer, no-progress guard, step budget), grounding | `src/movie_finder/agent/` |
| Guarded text-to-SQL over metadata | `src/movie_finder/data/moviedb.py`, `src/movie_finder/agent/sql_tool.py` |
| Generate-then-verify (`check_title`) | `src/movie_finder/titles.py`, `src/movie_finder/agent/tools.py` |
| Served interface: FastAPI + Server-Sent Events, Thai UI | `src/movie_finder/api/` |

## Setup

Needs Python 3.10+ and the class proxy's API key and base URL. Commands are for Windows
PowerShell; on macOS/Linux use `source .venv/bin/activate` and `cp` instead of `copy`.

```powershell
git clone <this repo>
cd movie-finder
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Open `.env` and set `OPENAI_API_KEY` and `OPENAI_BASE_URL` (the base URL ends in `/v1`).
Leave the other lines as they are; they are the tuned defaults.

```powershell
# 1. check the proxy: Thai chat, JSON output, tool calling -> three [PASS] lines
python scripts\smoke_test_llm.py

# 2. download + clean the corpus -> data\processed\movies.jsonl (18,432 films)
python scripts\fetch_data.py
#    if the Kaggle download fails: download wiki_movie_plots_deduped.csv from the link above, then
#    python scripts\fetch_data.py --csv path\to\wiki_movie_plots_deduped.csv

# 3. chunk + embed the plots -> data\index\ (downloads the embedding model once; slow on CPU)
python scripts\build_index.py

# 4. run the tests (no network or API key needed)
pytest -q
```

## Run the app

```powershell
uvicorn movie_finder.api.app:app --app-dir src --port 8000
```

Open <http://localhost:8000>. The first request after start-up takes a few extra seconds while
the models load. Stop the server with `Ctrl+C`.

Try for example:

- `ผู้ชายตื่นขึ้นมาแล้วเจอวันเดิมซ้ำไปเรื่อย ๆ ในเมืองเล็ก ๆ ช่วงหน้าหนาว` → *Groundhog Day*
- `the Mad Max movie with Mel Gibson where they drive across the desert in a war rig with escaped wives`
  → *Mad Max: Fury Road* (the actor is misremembered; watch the SQL step and the "outside" matches)
- `หนังไทยเรื่องพี่มากพระโขนง` → not in the database (it has no Thai films)

| endpoint | what it does |
|---|---|
| `GET /` | demo page: the agent's steps stream in, then the answer with its cited passage |
| `GET /health` | corpus, index and model info |
| `POST /search` | retrieval only, no LLM: `{"query": "...", "k": 5, "mode": "hybrid"}` (`mode`: `dense` / `bm25` / `hybrid`) |
| `POST /find` | the full agent, one JSON reply: `{"description": "...", "history": []}` |
| `POST /find/stream` | the same as Server-Sent Events: `tool_call`, `tool_result`, `final` |

## Evaluate

Gold sets are JSONL files: a description, the film's title + year, and tags. `"title": null`
marks a film that is **not** in the corpus, where the right answer is "not found".

| file | queries | purpose |
|---|---|---|
| `eval/queries.jsonl` | 42 answerable + 7 not in corpus | main set, used for every decision |
| `eval/heldout.jsonl` | 16 + 6 | written by a separate agent after tuning; run once |
| `eval/heldout2.jsonl` | 14 + 10 | written after `check_title` was designed, mostly visual memories; run once |

```powershell
python eval\run_eval.py                         # retrieval only: BM25 vs dense vs hybrid + min_score sweep (no LLM)
python eval\run_eval.py --rewrite --agent       # + LLM-rewritten queries + the full agent vs LLM-only (~10 min)
python eval\run_eval.py --gold eval\heldout2.jsonl --out eval\results_heldout2 --rewrite --agent
```

Each run writes `report.md` (all tables), `per_query.md` (top-5 films for every query) and
`results.json` to its `--out` folder (default `eval\results\`). The committed reports are the
final runs: [main](eval/results/report.md), [held-out 1](eval/results_heldout/report.md),
[held-out 2](eval/results_heldout2/report.md).

Metrics: recall@1/5/10 with a 95% bootstrap CI, MRR@10, nDCG@10 and median rank for
retrieval; for the agent, top-1 accuracy vs. the LLM-only baseline, precision of "found",
"not found" rate on films outside the corpus, where answers are lost (retrieval vs. selection),
and how often each tool was used.

<details>
<summary>More tools and experiments</summary>

```powershell
python scripts\find_movie.py "groundhog"                        # look up a title in the corpus
python scripts\search.py "a guy relives the same day" --rewrite  # top-5 per retrieval mode for any description
python scripts\check_dense.py                                   # sanity check: a passage finds its own film
python scripts\replay_searches.py eval\results\results.json     # replay the agent's searches without the LLM

python eval\run_eval.py --rewrite --multi                       # + 3 phrasings per query fused with RRF
python eval\run_eval.py --rewrite --rerank cross-encoder/ms-marco-MiniLM-L-6-v2   # + cross-encoder rerank rows

# other indexes, compared side by side in one report
python scripts\build_index.py --chunk-words 100 --chunk-overlap 25 --out data\index_c100
python scripts\build_index.py --embed-model BAAI/bge-small-en-v1.5 --out data\index_bge
python eval\run_eval.py --index data\index data\index_c100 data\index_bge --rewrite
```

Each index records its embedding model in `meta.json`, and the app always embeds queries with
that model. Similarity scales differ between models, so re-pick `MIN_SCORE` from the sweep
when switching.
</details>

## Configuration

All settings come from `.env` (see [`.env.example`](.env.example) for what each one does and
why it has its value). Only `OPENAI_API_KEY` and `OPENAI_BASE_URL` are required. Secrets are
never committed: `.env` is git-ignored.

## Project layout

```
movie-finder/
├── src/movie_finder/
│   ├── config.py          settings from .env (the only place that reads the environment)
│   ├── llm.py             the only module that calls the LLM proxy; JSON parsing + repair
│   ├── schemas.py         Pydantic models: corpus, tool arguments, final answer, API
│   ├── titles.py          title normalisation + lookup (check_title, gold labels)
│   ├── evaluation.py      gold-set loading and metrics
│   ├── data/              corpus cleaning + chunking; read-only metadata DB with SQL guards
│   ├── retrieval/         embedder, vector store, BM25, hybrid retriever, query rewrite, reranker
│   ├── agent/             tools (allow-list, validation) and the agent loop (stop conditions, grounding)
│   └── api/               FastAPI app + static/index.html (the demo page)
├── scripts/               fetch_data, build_index, smoke_test_llm, search, find_movie, check_dense, replay_searches
├── eval/                  gold sets, run_eval.py, committed results
├── docs/experiments.md    every experiment with before/after numbers, error analysis, limitations
├── tests/                 unit tests with fixtures and a fake LLM (no network, no key)
└── data/                  raw / processed / index -- git-ignored, rebuilt by the scripts
```

Layering: `api → agent → retrieval / data / llm → config`; lower layers never import upper ones.

## Limitations

The corpus stops at 2017 and has no Thai, French, Spanish or Swedish films. Plot summaries
describe events, not images, so a purely visual memory is found only if the model already
knows the film. The test sets are small (one query moves held-out accuracy by 6–7 points), and
the main set was used for tuning. See [`docs/experiments.md`](docs/experiments.md).
