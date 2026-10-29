# Evaluation report

_Generated 2026-10-29 05:10 UTC — 16 answerable + 6 unanswerable queries from `eval/queries.jsonl`._

LLM: `gemma-4-E4B-it` · embeddings: `intfloat/multilingual-e5-small` · temperature 0

## 1. Retrieval (answerable queries)

Each query has one correct movie (or a few accepted versions), so recall@k = hit rate@k. Precision@k is not reported: with one relevant item it is just recall@k / k. The 95% CI is a bootstrap over queries — with ~30 queries, one query moves recall by ~3 points, so differences inside the CI are not reliable.

| index | chunk words | mode | recall@1 | recall@5 (95% CI) | recall@10 | MRR@10 | nDCG@10 | median rank |
|---|---|---|---|---|---|---|---|---|
| `data/index` | 200 | bm25 (baseline) | 56.2% | 56.2% (31.2%–81.2%) | 56.2% | 0.562 | 0.562 | 1 |
| `data/index` | 200 | dense | 12.5% | 43.8% (18.8%–68.8%) | 50.0% | 0.219 | 0.285 | 10 |
| `data/index` | 200 | hybrid | 43.8% | 50.0% (25.0%–75.0%) | 50.0% | 0.450 | 0.462 | 8 |
| `data/index` | 200 | bm25+rewrite | 68.8% | 81.2% (62.5%–100.0%) | 81.2% | 0.721 | 0.743 | 1 |
| `data/index` | 200 | dense+rewrite | 37.5% | 50.0% (25.0%–75.0%) | 56.2% | 0.438 | 0.468 | 4.5 |
| `data/index` | 200 | hybrid+rewrite | 56.2% | 75.0% (50.0%–93.8%) | 81.2% | 0.638 | 0.680 | 1 |

### recall@5 by query tag (`data/index`)

| tag | n | bm25 | dense | hybrid | bm25+rewrite | dense+rewrite | hybrid+rewrite |
|---|---|---|---|---|---|---|---|
| en | 12 | 75.0% | 58.3% | 66.7% | 75.0% | 50.0% | 75.0% |
| famous | 10 | 40.0% | 30.0% | 30.0% | 70.0% | 30.0% | 60.0% |
| metadata | 3 | 66.7% | 66.7% | 66.7% | 66.7% | 33.3% | 66.7% |
| obscure | 6 | 83.3% | 66.7% | 83.3% | 100.0% | 83.3% | 100.0% |
| th | 4 | 0.0% | 0.0% | 0.0% | 100.0% | 50.0% | 75.0% |
| vague | 2 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| wrong_detail | 1 | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% | 100.0% |

Rows marked `+rewrite` search the LLM-rewritten query (`eval/results/rewrites.json`). Rewrites that named the gold movie: none.

## 2. `min_score` floor on the dense similarity

Each pair: recall@5 kept on answerable queries / share of unanswerable queries that return nothing.

| min_score | dense, raw: recall@5 | dense, raw: nothing | dense, rewritten: recall@5 | dense, rewritten: nothing | **hybrid, rewritten (as the app runs)**: recall@5 | **hybrid, rewritten (as the app runs)**: nothing |
|---|---|---|---|---|---|---|
| 0.00 | 43.8% | 0.0% | 50.0% | 0.0% | 75.0% | 0.0% |
| 0.70 | 43.8% | 0.0% | 50.0% | 0.0% | 75.0% | 0.0% |
| 0.74 | 43.8% | 0.0% | 50.0% | 0.0% | 75.0% | 0.0% |
| 0.76 | 43.8% | 0.0% | 50.0% | 0.0% | 75.0% | 0.0% |
| 0.78 | 43.8% | 0.0% | 50.0% | 0.0% | 75.0% | 0.0% |
| 0.80 | 43.8% | 0.0% | 50.0% | 0.0% | 75.0% | 0.0% |
| 0.82 | 43.8% | 0.0% | 50.0% | 0.0% | 56.2% | 0.0% |
| 0.84 | 31.2% | 66.7% | 31.2% | 33.3% | 31.2% | 33.3% |
| 0.86 | 6.2% | 83.3% | 6.2% | 83.3% | 6.2% | 83.3% |

## 3. End-to-end: agent (hybrid retrieval) vs. LLM-only (no retrieval)

| system | top-1 accuracy (answerable) | says not found (unanswerable) |
|---|---|---|
| LLM-only baseline | 50.0% | 0.0% |
| Agent + retrieval | 68.8% | 16.7% |

Average 2.4 model turns and 4.1s per query.

### Where the agent loses answers (answerable queries)

| step | share of answerable queries |
|---|---|
| correct movie appeared in any of the agent's searches | 68.8% |
| correct movie among the agent's ≤3 candidates | 68.8% |
| correct movie ranked first by the agent | 68.8% |

100% minus the first row is a retrieval error (the right movie was never retrieved); the drop from the first row to the last is a selection error (retrieved, but not ranked first).

Precision of "found": when the agent says it found the movie, it is right 88.9% of the time (9 "found" answers).

| outcome | answerable | unanswerable |
|---|---|---|
| found, correct | 8 | 0 |
| not sure, correct on top | 3 | 0 |
| not sure, wrong on top | 4 | 0 |
| found, WRONG | 0 | 1 |
| not sure | 0 | 4 |
| said not found | 1 | 1 |

How runs ended — final_answer: 22. Queries where the agent used the metadata database (text-to-SQL): 4.

### Top-1 accuracy by query tag (answerable)

| tag | n | agent | LLM-only |
|---|---|---|---|
| en | 12 | 66.7% | 41.7% |
| famous | 10 | 50.0% | 80.0% |
| metadata | 3 | 66.7% | 33.3% |
| obscure | 6 | 100.0% | 0.0% |
| th | 4 | 75.0% | 75.0% |
| vague | 2 | 0.0% | 100.0% |
| wrong_detail | 1 | 100.0% | 100.0% |

## Dense misses (rank ≠ 1) — for error analysis

| id | query | gold | rank | dense top-1 |
|---|---|---|---|---|
| h01 | guy goes to prison for killing his wife but he didn't do it. spends like 20 year | The Shawshank Redemption | >10 | Sherlock Toms (2017) |
| h02 | การ์ตูนที่พ่อปลาการ์ตูนออกตามหาลูกที่โดนนักดำน้ำจับไปไว้ในตู้ปลาที่คลินิกหมอฟัน  | Finding Nemo | >10 | Bay, TheThe Bay (2012) |
| h03 | 80s action movie, bruce willis running around barefoot in an office tower during | Die Hard | >10 | Go Goa Gone (2013) |
| h04 | FBI trainee has to interview a creepy cannibal psychiatrist through glass to cat | The Silence of the Lambs | 2 | That Darn Cat (1997) |
| h05 | หนังที่ตำรวจไปสืบคดีคนไข้หายบนเกาะที่เป็นโรงพยาบาลจิตเวช บรรยากาศหลอนๆ ตอนจบหักม | Shutter Island | >10 | Don't Forget me (2016) |
| h06 | the one where the guy with insomnia goes to cancer support groups to feel someth | Fight Club | 4 | Christmas with the Dead (2012) |
| h07 | alien gets left behind and a kid hides it in his closet, bikes flying in front o | E.T. the Extra-Terrestrial | 9 | First on the Moon (2005) |
| h08 | เด็กผู้ชายอินเดียติดอยู่บนเรือชูชีพกลางทะเลกับเสือเบงกอล ภาพสวยมาก ตอนท้ายมีเล่า | Life of Pi | >10 | Bicycle Kick (2013) |
| h09 | space movie where julia roberts is an astronaut and debris smashes their shuttle | Gravity | 5 | First on the Moon (2005) |
| h10 | cute little robot alone collecting garbage, barely any talking for the first hal | WALL-E | >10 | Adventures of Omanakuttan (2017) |
| h11 | Wong Kar-wai movie, a cop keeps buying cans of pineapple that expire on May 1 af | Chungking Express | 4 | The Quiet Room (1996) |
| h13 | เด็กผู้ชายเรียนไม่ได้ อ่านหนังสือตัวอักษรเต้นไปมา พ่อเลยส่งไปโรงเรียนประจำ แล้วม | Taare Zameen Par | >10 | The Road Home (1999) |
| h15 | teenage kids on a london council estate fighting off black furry aliens with glo | Attack the Block | >10 | Flesheater (1988) |
| h18 | truck driver cowboy guy helps a widow make the perfect ramen for her noodle shop | Tampopo | 5 | CR No: 89 (2015) |
