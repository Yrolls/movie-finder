# Experiment log and final numbers

All numbers come from the committed reports in `eval/` (each experiment is a separate git
commit, so every row below can be reproduced from history). Model: `gemma-4-E4B-it` through
the class proxy, temperature 0. Corpus: Wikipedia Movie Plots, films 1980–2017,
18,432 films → 59,558 chunks of 200 words (50 overlap).

## Final results

| set | written by | answerable / not in corpus | agent top-1 | LLM-only top-1 | agent − LLM-only (95% CI) | precision of "found" |
|---|---|---|---|---|---|---|
| main (`eval/queries.jsonl`) | me, used for all tuning | 42 / 7 | **88.1%** (78.6–97.6) | 71.4% | +16.7 pts (+7.1 … +28.6) | 96.9% (31/32) |
| held-out 1 (`eval/heldout.jsonl`) | a separate agent, after tuning | 16 / 6 | **68.8%** (43.8–87.5) | 50.0% | +18.8 pts (−18.8 … +56.3) | 100% (8/8) |
| held-out 2 (`eval/heldout2.jsonl`) | a separate agent, after `check_title` was designed | 14 / 10 | **71.4%** (50.0–92.9) | 35.7% | +35.7 pts (0.0 … +64.3) | 87.5% (7/8)* |

CIs are a paired bootstrap over queries (5,000 resamples). Held-out 1 was run with the
configuration of 29 Oct 08:05 (before Thai answers, `check_title` and the second look);
held-out 2 with the final one. Each held-out set was run once with a frozen configuration.

\* The one wrong "found" on held-out 2 is k09: the user describes *Psycho* (1960), which is
outside 1980–2017, and the agent answered *Psycho* (1998), the shot-for-shot remake that is in
the corpus. Counted as wrong here; arguably a label ambiguity.

Who gets which query right (answerable): main set 30 both, 7 agent only, 0 LLM-only only;
held-out 1: 5 / 6 / 3; held-out 2: 4 / 6 / 1.

### Retrieval on the main set (42 answerable queries, final index)

| mode | recall@1 | recall@5 (95% CI) | recall@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|
| BM25 (named baseline) | 31.0% | 42.9% (28.6–59.5) | 45.2% | 0.362 | 0.384 |
| dense (multilingual-e5-small) | 14.3% | 28.6% (16.7–42.9) | 33.3% | 0.201 | 0.233 |
| hybrid (BM25 + dense, RRF) | 33.3% | 50.0% (35.7–64.3) | 52.4% | 0.393 | 0.424 |
| hybrid + LLM rewrite (**the app**) | 50.0% | **76.2%** (64.3–88.1) | 78.6% | 0.596 | 0.642 |

Thai queries (15): recall@5 0% for every mode on the raw text → 80% after the LLM rewrites
them into an English plot summary.

### Final agent behaviour (main set)

- correct film appeared in some tool result: 92.9%; among the ≤3 candidates: 90.5%; ranked first: 88.1%
- tools used: text-to-SQL on 7 queries, `check_title` on 16 (1 correct answer came only from it;
  2 on held-out 2), second look asked 16 times
- average 3.6 model turns, 10.1 s per query; 23 invalid JSON answers repaired, 2 runs ended as
  `invalid_output`
- 0 wrong "found" on answerable queries. The only wrong "found" is q34: the user describes
  *Parasite* (2019, not in the corpus); the model named "Parasite", `check_title` matched the
  unrelated *Parasite* (1982) by title, and the agent accepted it -- a same-title false positive.

## What was tried, in order

| # | change | measured on | before → after | kept? |
|---|---|---|---|---|
| 1 | first retrieval eval | main (32 q) | BM25 R@5 46.9%, dense 31.2%, hybrid 46.9%; Thai 0% everywhere | baseline |
| 2 | LLM query rewrite (Thai/vague → English plot) | main (32 q) | hybrid R@5 46.9% → 75.0%; Thai 0% → 76.9% | ✅ |
| 3 | multi-query (3 phrasings, RRF) | main (32 q) | hybrid+rewrite 75.0% → 68.8% | ❌ (single rewrite kept) |
| 4 | `MIN_SCORE` 0.82 from a dense-only sweep | main | halved hybrid recall in the app → re-swept under hybrid, 0.78 | 0.78 |
| 5 | first end-to-end agent | main (32 q) | agent 59.4% vs LLM-only 68.8% (lost!) | — |
| 6 | extended metrics (nDCG, median rank, CIs, error breakdown) | — | — | ✅ |
| 7 | text-to-SQL over metadata + no-progress guard; set grown to 49 q | main (42 q) | agent 73.8% vs 71.4% | ✅ |
| 8 | relax empty SQL once, show plot matches outside the SQL set, `get_movie` budget | main | 73.8% (precision 93.3% → 92.0%) | ✅ |
| 9 | model sees the hybrid **rank**, not the dense similarity | main | 73.8% → **85.7%**, precision 100% | ✅ |
| 10 | held-out 1 (run once) | held-out 1 | agent 68.8% vs LLM-only 50.0% | — |
| 11 | fix: hybrid fusion dropped films found only by BM25 | main | hybrid+rewrite R@5 73.8% → 76.2% | ✅ (bug fix) |
| 12 | cross-encoder rerank (ms-marco-MiniLM-L-6-v2) | main | R@5 76.2% → 76.2%, MRR 0.596 → 0.641 | ❌ (no recall gain, slower) |
| 13 | English embedder bge-small-en-v1.5 (+rerank) | main | best retrieval row 78.6% → 81.0% (inside CI); agent 85.7% → 83.3% | ❌ reverted |
| 14 | answers in Thai; explicit "not sure" / "not in the database" | UI | — | ✅ |
| 15 | `check_title`: model names films, corpus verifies (for "how it looked" memories) | main | 85.7% → 78.6%; tool never called | kept, needed #16 |
| 16 | "second look": one forced `check_title` or re-search before settling for not sure | main | 78.6% → 83.3%; `check_title` used on 17 q | ✅ |
| 17 | JSON repairs no longer consume the tool-step budget | main | 83.3% → **88.1%**; runs cut off by max_steps 11 → 0 | ✅ (final) |
| 18 | held-out 2 (run once, frozen config) | held-out 2 | agent 71.4% vs LLM-only 35.7% | — |

Diagnosis behind #16: `scripts/replay_searches.py` re-ran the agent's recorded searches
without the LLM. Old and new fusion code returned the gold film for exactly the same searches
(29/37 and 31/36), and the `MIN_SCORE` floor removed none (0 / 0.74 / 0.76 / 0.78 identical) —
the drop in #15 came from the agent's wording plus never retrying, not from retrieval code.

Not acted on (found late, inside the CI, out of time): after the fusion fix (#11),
hybrid + multi-query reached R@5 83.3% vs 76.2% for the single rewrite.

## Error analysis (final configuration)

Main set, 5 misses: q24 *The Station Agent*, q29 *3 Idiots*, q30 *Departures* — obscure films
that neither the search nor the model found; q06 *Back to the Future* — retrieved, but the
agent preferred a film it had named itself; q38 *It* — the model's answer stayed invalid JSON
after 2 repairs, although *It* was in its search results.

Held-out 2, 4 misses: k01 *The Shining*, k04 *Spirited Away*, k10 *Back to the Future*,
k16 *Pan's Labyrinth*. Three are pure "what it looked like" memories (blue-dress twins and
hexagon carpet, the masked spirit, the pale man with eyes in his palms). Plot summaries do not
describe these images, and the 4B model guessed wrong titles (*The Innocents*, *Dhayam*,
*About Time*), so `check_title` had nothing correct to verify. The grounding held: none of the
wrong guesses became a "found".

## Limitations

- Small sets: one query moves main-set accuracy by 2.4 points and held-out by 6–7 points; the
  held-out gains over LLM-only have wide CIs.
- The main set was used for every decision, so 88.1% is optimistic; the held-out sets
  (68.8%, 71.4%) are the honest estimate.
- Held-out 1 was opened once for failure analysis (two labels fixed to "not in corpus"), and
  its failures motivated `check_title`; held-out 2 was written afterwards to test that.
- Corpus: no Thai, French, Spanish or Swedish films; nothing after 2017; some films listed twice.
- Plot summaries describe events, not images, so purely visual memories depend on the model
  already knowing the film; `gemma-4-E4B-it` often does not.
- The 4B model often returns malformed JSON (23 repairs on 49 queries).
- `check_title` matches by title only, so a remembered film that shares its title with an older
  one in the corpus can be confirmed wrongly (q34 *Parasite*); the year/plot check is left to
  the model.
- "Not found" on films outside the corpus is weak (43% main, 30% held-out 2): the agent
  usually says "not sure" with a follow-up question instead, never a confident wrong answer
  except q34 / k09.

## Next steps

- A larger model for `check_title` (knowledge) and for valid JSON (fewer repairs).
- Have `check_title` refuse a "found" when the cited passage's similarity is low, to stop
  same-title false positives such as q34.
- Hybrid + multi-query in the app (83.3% R@5 after the fusion fix), measured on a new held-out set.
- Enrich the corpus with descriptions of how films look (posters, taglines, full Wikipedia
  articles) and add Thai films.
