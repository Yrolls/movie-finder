# Evaluation report

_Generated 2026-10-29 09:32 UTC — 14 answerable + 10 unanswerable queries from `eval/heldout2.jsonl`._

LLM: `gemma-4-E4B-it` · embeddings: `intfloat/multilingual-e5-small` (`data/index`) · reranker: `none` · temperature 0

## 1. Retrieval (answerable queries)

Each query has one correct movie (or a few accepted versions), so recall@k = hit rate@k. Precision@k is not reported: with one relevant item it is just recall@k / k. The 95% CI is a bootstrap over queries — with ~30 queries, one query moves recall by ~3 points, so differences inside the CI are not reliable.

| index | chunk words | mode | recall@1 | recall@5 (95% CI) | recall@10 | MRR@10 | nDCG@10 | median rank |
|---|---|---|---|---|---|---|---|---|
| `data/index` | 200 | bm25 (baseline) | 50.0% | 50.0% (21.4%–78.6%) | 50.0% | 0.500 | 0.500 | 6 |
| `data/index` | 200 | dense | 14.3% | 14.3% (0.0%–35.7%) | 28.6% | 0.163 | 0.190 | >10 |
| `data/index` | 200 | hybrid | 35.7% | 50.0% (21.4%–78.6%) | 57.1% | 0.429 | 0.463 | 4.5 |
| `data/index` | 200 | bm25+rewrite | 50.0% | 71.4% (50.0%–92.9%) | 78.6% | 0.591 | 0.638 | 1.5 |
| `data/index` | 200 | dense+rewrite | 14.3% | 14.3% (0.0%–35.7%) | 21.4% | 0.150 | 0.164 | >10 |
| `data/index` | 200 | hybrid+rewrite | 42.9% | 71.4% (42.9%–92.9%) | 71.4% | 0.514 | 0.563 | 3 |

### recall@5 by query tag (`data/index`)

| tag | n | bm25 | dense | hybrid | bm25+rewrite | dense+rewrite | hybrid+rewrite |
|---|---|---|---|---|---|---|---|
| en | 10 | 70.0% | 20.0% | 70.0% | 80.0% | 20.0% | 80.0% |
| famous | 9 | 44.4% | 0.0% | 44.4% | 55.6% | 0.0% | 55.6% |
| metadata | 3 | 66.7% | 33.3% | 66.7% | 66.7% | 33.3% | 66.7% |
| obscure | 5 | 60.0% | 40.0% | 60.0% | 100.0% | 40.0% | 100.0% |
| th | 4 | 0.0% | 0.0% | 0.0% | 50.0% | 0.0% | 50.0% |
| vague | 2 | 50.0% | 50.0% | 50.0% | 100.0% | 50.0% | 100.0% |
| visual | 8 | 50.0% | 0.0% | 50.0% | 75.0% | 0.0% | 75.0% |
| wrong_detail | 2 | 50.0% | 0.0% | 50.0% | 50.0% | 0.0% | 50.0% |

Rows marked `+rewrite` search the LLM-rewritten query (`eval/results_heldout2/rewrites.json`). Rewrites that named the gold movie: none.

## 2. `min_score` floor on the dense similarity

Each pair: recall@5 kept on answerable queries / share of unanswerable queries that return nothing.

| min_score | dense, raw: recall@5 | dense, raw: nothing | dense, rewritten: recall@5 | dense, rewritten: nothing | **hybrid, rewritten (as the app runs)**: recall@5 | **hybrid, rewritten (as the app runs)**: nothing |
|---|---|---|---|---|---|---|
| 0.00 | 14.3% | 0.0% | 14.3% | 0.0% | 71.4% | 0.0% |
| 0.40 | 14.3% | 0.0% | 14.3% | 0.0% | 71.4% | 0.0% |
| 0.50 | 14.3% | 0.0% | 14.3% | 0.0% | 71.4% | 0.0% |
| 0.55 | 14.3% | 0.0% | 14.3% | 0.0% | 71.4% | 0.0% |
| 0.60 | 14.3% | 0.0% | 14.3% | 0.0% | 71.4% | 0.0% |
| 0.65 | 14.3% | 0.0% | 14.3% | 0.0% | 71.4% | 0.0% |
| 0.70 | 14.3% | 0.0% | 14.3% | 0.0% | 71.4% | 0.0% |
| 0.74 | 14.3% | 0.0% | 14.3% | 0.0% | 71.4% | 0.0% |
| 0.76 | 14.3% | 0.0% | 14.3% | 0.0% | 71.4% | 0.0% |
| 0.78 | 14.3% | 0.0% | 14.3% | 0.0% | 71.4% | 0.0% |
| 0.80 | 14.3% | 0.0% | 14.3% | 0.0% | 64.3% | 0.0% |
| 0.82 | 14.3% | 0.0% | 14.3% | 0.0% | 28.6% | 10.0% |
| 0.84 | 14.3% | 50.0% | 14.3% | 50.0% | 14.3% | 50.0% |
| 0.86 | 0.0% | 90.0% | 0.0% | 100.0% | 0.0% | 100.0% |

## 3. End-to-end: agent (hybrid retrieval) vs. LLM-only (no retrieval)

| system | top-1 accuracy (answerable) | says not found (unanswerable) |
|---|---|---|
| LLM-only baseline | 35.7% | 20.0% |
| Agent + retrieval | 71.4% | 30.0% |

Average 4.1 model turns and 9.9s per query.

### Where the agent loses answers (answerable queries)

| step | share of answerable queries |
|---|---|
| correct movie appeared in any tool result (searches, SQL, check_title) | 71.4% |
| correct movie among the agent's ≤3 candidates | 71.4% |
| correct movie ranked first by the agent | 71.4% |

100% minus the first row is a retrieval error (the right movie was never retrieved); the drop from the first row to the last is a selection error (retrieved, but not ranked first).

Precision of "found": when the agent says it found the movie, it is right 87.5% of the time (8 "found" answers).

| outcome | answerable | unanswerable |
|---|---|---|
| found, correct | 7 | 0 |
| not sure, correct on top | 3 | 0 |
| not sure, wrong on top | 3 | 0 |
| found, WRONG | 0 | 1 |
| not sure | 0 | 6 |
| said not found | 1 | 3 |

How runs ended — final_answer: 20, invalid_output: 3, max_steps: 1. Queries where the agent used the metadata database (text-to-SQL): 5.

`check_title` (the model names films it knows, the database verifies them): used on 16 queries; correct answers that came only from it: 2; right film reached only through it: 14.3% of answerable queries. Second looks asked: 8; invalid answers repaired: 12.

### Top-1 accuracy by query tag (answerable)

| tag | n | agent | LLM-only |
|---|---|---|---|
| en | 10 | 80.0% | 40.0% |
| famous | 9 | 55.6% | 44.4% |
| metadata | 3 | 66.7% | 66.7% |
| obscure | 5 | 100.0% | 20.0% |
| th | 4 | 50.0% | 25.0% |
| vague | 2 | 100.0% | 0.0% |
| visual | 8 | 62.5% | 37.5% |
| wrong_detail | 2 | 50.0% | 50.0% |

## Dense misses (rank ≠ 1) — for error analysis

| id | query | gold | rank | dense top-1 |
|---|---|---|---|---|
| k01 | creepy old hotel movie, i mostly remember the two little girls in blue dresses s | The Shining | >10 | The Quiet Room (1996) |
| k04 | การ์ตูนญี่ปุ่นที่มีผีตัวดำๆใส่หน้ากากขาว ไม่พูด ชอบยื่นทองให้คน แล้วพ่อแม่นางเอก | Spirited Away | >10 | April Fools (2015) |
| k06 | that james cameron dinosaur movie from the 80s, the one with the glass of water  | Jurassic Park | >10 | Tuck Everlasting (1981) |
| k07 | อนิเมะที่มีขบวนพาเหรดประหลาดมาก ตู้เย็น ตุ๊กตา กบเป่าแตร เดินกันเต็มถนน เกี่ยวกั | Paprika | >10 | Dragon Head (2003) |
| k10 | หนังเด็กวัยรุ่นนั่งรถย้อนเวลากลับไปตอนพ่อแม่ยังเรียนอยู่ แล้วแม่ดันมาชอบตัวเองแท | Back to the Future | >10 | Nenu Local (2017) |
| k11 | pale guy with crazy black hair and blades instead of fingers who lives in a cast | Edward Scissorhands | >10 | Ant Story (2014) |
| k16 | almodovar movie set in the spanish civil war with a girl and a faun, theres a te | Pan's Labyrinth | >10 | Unrest (2006) |
| k18 | woman goes on a sailing trip with friends, boat flips, they get on an empty crui | Triangle | >10 | Sally Marshall Is Not an Alien (1999) |
| k19 | การ์ตูนที่ตาแก่ผูกลูกโป่งเป็นพันๆลูกให้บ้านลอยไป แล้วมีเด็กลูกเสืออ้วนๆติดไปด้วย | Up | >10 | Missing (2016) |
| k20 | fbi trainee has to interview a creepy cannibal psychiatrist in a glass cell to c | The Silence of the Lambs | 7 | Dhayam (2017) |
| k23 | gorgeous looking movie, injured stuntman in a 1920s hospital tells a little girl | The Fall | 7 | Antarmahal (2005) |
| k24 | guy in a long black coat gets offered a red pill or a blue pill and finds out th | The Matrix | >10 | Newton (2017) |
