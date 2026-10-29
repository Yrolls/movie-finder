# Evaluation report

_Generated 2026-10-29 04:27 UTC — 42 answerable + 7 unanswerable queries from `eval/queries.jsonl`._

LLM: `gemma-4-E4B-it` · embeddings: `intfloat/multilingual-e5-small` · temperature 0

## 1. Retrieval (answerable queries)

Each query has one correct movie (or a few accepted versions), so recall@k = hit rate@k. Precision@k is not reported: with one relevant item it is just recall@k / k. The 95% CI is a bootstrap over queries — with ~30 queries, one query moves recall by ~3 points, so differences inside the CI are not reliable.

| index | chunk words | mode | recall@1 | recall@5 (95% CI) | recall@10 | MRR@10 | nDCG@10 | median rank |
|---|---|---|---|---|---|---|---|---|
| `data/index` | 200 | bm25 (baseline) | 31.0% | 42.9% (28.6%–59.5%) | 45.2% | 0.362 | 0.384 | >10 |
| `data/index` | 200 | dense | 14.3% | 28.6% (16.7%–42.9%) | 33.3% | 0.201 | 0.233 | >10 |
| `data/index` | 200 | hybrid | 33.3% | 50.0% (35.7%–64.3%) | 52.4% | 0.393 | 0.424 | 7 |
| `data/index` | 200 | bm25+rewrite | 35.7% | 61.9% (47.6%–76.2%) | 71.4% | 0.482 | 0.538 | 2 |
| `data/index` | 200 | dense+rewrite | 14.3% | 28.6% (16.7%–42.9%) | 40.5% | 0.211 | 0.256 | >10 |
| `data/index` | 200 | hybrid+rewrite | 50.0% | 73.8% (59.5%–85.7%) | 76.2% | 0.591 | 0.633 | 1.5 |
| `data/index` | 200 | bm25+multi | 42.9% | 71.4% (57.1%–83.3%) | 78.6% | 0.548 | 0.606 | 2 |
| `data/index` | 200 | dense+multi | 14.3% | 31.0% (16.7%–45.2%) | 42.9% | 0.217 | 0.266 | >10 |
| `data/index` | 200 | hybrid+multi | 50.0% | 69.0% (54.8%–83.3%) | 73.8% | 0.570 | 0.611 | 1.5 |

### recall@5 by query tag (`data/index`)

| tag | n | bm25 | dense | hybrid | bm25+rewrite | dense+rewrite | hybrid+rewrite | bm25+multi | dense+multi | hybrid+multi |
|---|---|---|---|---|---|---|---|---|---|---|
| confusable | 3 | 66.7% | 0.0% | 66.7% | 66.7% | 0.0% | 33.3% | 66.7% | 0.0% | 0.0% |
| en | 27 | 66.7% | 44.4% | 77.8% | 63.0% | 37.0% | 70.4% | 74.1% | 37.0% | 70.4% |
| famous | 34 | 38.2% | 29.4% | 50.0% | 64.7% | 29.4% | 76.5% | 70.6% | 32.4% | 73.5% |
| metadata | 10 | 30.0% | 20.0% | 60.0% | 60.0% | 20.0% | 70.0% | 70.0% | 40.0% | 70.0% |
| obscure | 8 | 62.5% | 25.0% | 50.0% | 50.0% | 25.0% | 62.5% | 75.0% | 25.0% | 50.0% |
| partial | 2 | 50.0% | 50.0% | 50.0% | 50.0% | 50.0% | 50.0% | 50.0% | 50.0% | 50.0% |
| plain | 24 | 41.7% | 37.5% | 41.7% | 62.5% | 37.5% | 79.2% | 70.8% | 33.3% | 75.0% |
| th | 15 | 0.0% | 0.0% | 0.0% | 60.0% | 13.3% | 80.0% | 66.7% | 20.0% | 66.7% |
| vague | 3 | 33.3% | 33.3% | 33.3% | 66.7% | 33.3% | 66.7% | 100.0% | 33.3% | 66.7% |
| wrong_detail | 3 | 100.0% | 0.0% | 100.0% | 66.7% | 0.0% | 100.0% | 100.0% | 33.3% | 100.0% |

Rows marked `+rewrite` search the LLM-rewritten query (`eval/results/rewrites.json`). Rewrites that named the gold movie: none.

Rows marked `+multi` search 3 LLM phrasings per query fused with RRF (`eval/results/rewrites_multi.json`). Phrasings that named the gold movie: none.

## 2. `min_score` floor on the dense similarity

Each pair: recall@5 kept on answerable queries / share of unanswerable queries that return nothing.

| min_score | dense, raw: recall@5 | dense, raw: nothing | dense, rewritten: recall@5 | dense, rewritten: nothing | **hybrid, rewritten (as the app runs)**: recall@5 | **hybrid, rewritten (as the app runs)**: nothing |
|---|---|---|---|---|---|---|
| 0.00 | 28.6% | 0.0% | 28.6% | 0.0% | 73.8% | 0.0% |
| 0.70 | 28.6% | 0.0% | 28.6% | 0.0% | 73.8% | 0.0% |
| 0.74 | 28.6% | 0.0% | 28.6% | 0.0% | 73.8% | 0.0% |
| 0.76 | 28.6% | 0.0% | 28.6% | 0.0% | 73.8% | 0.0% |
| 0.78 | 28.6% | 0.0% | 28.6% | 0.0% | 73.8% | 0.0% |
| 0.80 | 28.6% | 14.3% | 28.6% | 0.0% | 64.3% | 0.0% |
| 0.82 | 28.6% | 28.6% | 26.2% | 14.3% | 38.1% | 14.3% |
| 0.84 | 16.7% | 71.4% | 14.3% | 42.9% | 14.3% | 71.4% |
| 0.86 | 7.1% | 100.0% | 4.8% | 100.0% | 4.8% | 100.0% |

## 3. End-to-end: agent (hybrid retrieval) vs. LLM-only (no retrieval)

| system | top-1 accuracy (answerable) | says not found (unanswerable) |
|---|---|---|
| LLM-only baseline | 69.0% | 28.6% |
| Agent + retrieval | 73.8% | 42.9% |

Average 2.4 model turns and 4.3s per query.

### Where the agent loses answers (answerable queries)

| step | share of answerable queries |
|---|---|
| correct movie appeared in any of the agent's searches | 85.7% |
| correct movie among the agent's ≤3 candidates | 85.7% |
| correct movie ranked first by the agent | 73.8% |

100% minus the first row is a retrieval error (the right movie was never retrieved); the drop from the first row to the last is a selection error (retrieved, but not ranked first).

Precision of "found": when the agent says it found the movie, it is right 92.0% of the time (25 "found" answers).

| outcome | answerable | unanswerable |
|---|---|---|
| found, correct | 23 | 0 |
| not sure, correct on top | 8 | 0 |
| not sure, wrong on top | 9 | 0 |
| found, WRONG | 2 | 0 |
| not sure | 0 | 4 |
| said not found | 0 | 3 |

How runs ended — final_answer: 49. Queries where the agent used the metadata database (text-to-SQL): 7.

### Top-1 accuracy by query tag (answerable)

| tag | n | agent | LLM-only |
|---|---|---|---|
| confusable | 3 | 100.0% | 100.0% |
| en | 27 | 92.6% | 74.1% |
| famous | 34 | 76.5% | 76.5% |
| metadata | 10 | 100.0% | 80.0% |
| obscure | 8 | 62.5% | 37.5% |
| partial | 2 | 50.0% | 50.0% |
| plain | 24 | 66.7% | 62.5% |
| th | 15 | 40.0% | 60.0% |
| vague | 3 | 66.7% | 66.7% |
| wrong_detail | 3 | 66.7% | 100.0% |

## Dense misses (rank ≠ 1) — for error analysis

| id | query | gold | rank | dense top-1 |
|---|---|---|---|---|
| q01 | ผู้ชายคนหนึ่งตื่นขึ้นมาแล้วเจอวันเดิมซ้ำไปเรื่อย ๆ ในเมืองเล็ก ๆ ช่วงหน้าหนาว เข | Groundhog Day | >10 | Goodbye Promise (2012) |
| q02 | a team of thieves breaks into a rich heir's mind while he sleeps to plant an ide | Inception | >10 | Crazy Stone (2006) |
| q03 | guy can't make new memories so he tattoos notes on his body and uses polaroid ph | Memento | 4 | I.C.U. (2009) |
| q04 | ชายคนหนึ่งค่อย ๆ รู้ว่าชีวิตทั้งชีวิตของเขาเป็นรายการทีวี ทุกคนรอบตัวเป็นนักแสดง | The Truman Show | >10 | Khiladi 420 (2000) |
| q05 | the one where the guy is stuck on an island after a plane crash and his only fri | Cast Away | 2 | Nim's Island (2008) |
| q06 | วัยรุ่นย้อนเวลากลับไปด้วยรถสปอร์ต แล้วเผลอทำให้พ่อแม่ตัวเองไม่ได้รักกัน ต้องแก้ใ | Back to the Future | >10 | Missing (2016) |
| q07 | a child psychologist tries to help a scared little boy who says he can see ghost | The Sixth Sense | 5 | Darna Zaroori Hai (2006) |
| q08 | เด็กผู้ชายขอพรกับตู้เกมแล้วตื่นมาตัวโตเป็นผู้ใหญ่ ได้ไปทำงานบริษัทของเล่น | Big | >10 | Buddies in India (2017) |
| q10 | a man from eastern europe gets stuck living inside a new york airport for months | The Terminal | 5 | Spiders 3D (2013) |
| q11 | คุณลุงแก่ ๆ ผูกลูกโป่งเป็นพันลูกกับบ้านแล้วลอยไปอเมริกาใต้ มีเด็กลูกเสือติดไปด้ว | Up | >10 | Missing (2016) |
| q12 | หนูที่อยากเป็นเชฟ แอบควบคุมเด็กหนุ่มในครัวร้านอาหารที่ปารีสด้วยการดึงผมใต้หมวก | Ratatouille | >10 | Love Khichdi (2009) |
| q14 | a boy is accidentally left home alone during easter holiday and fights off two b | Home Alone | >10 | Look Who's Talking Too (1990) |
| q15 | soldier fighting aliens keeps dying and waking up on the same day before the bat | Edge of Tomorrow | >10 | Alice's Birthday (2009) |
| q16 | a man wakes up in someone else's body on a commuter train and keeps reliving the | Source Code | >10 | Third Contact (2011) |
| q17 | ผู้ชายเหงา ๆ ที่เพิ่งหย่า ตกหลุมรักระบบปฏิบัติการ AI ในคอมพิวเตอร์ที่มีแค่เสียงผ | Her | 6 | Queen of the Night (2013) |
| q18 | young programmer is invited to a billionaire's remote house to test whether a fe | Ex Machina | 2 | I Origins (2014) |
| q20 | นักวิทยาศาสตร์ผู้หญิงที่ฟังสัญญาณจากอวกาศ ได้รับข้อความจากดาวดวงอื่นที่มีแบบแปลน | Contact | >10 | Premam (2016) |
| q23 | ผู้ชายขี้อายพาตุ๊กตาขนาดเท่าคนจริงมาแนะนำว่าเป็นแฟน แล้วคนทั้งเมืองก็เล่นตามไปด้ | Lars and the Real Girl | >10 | Finding Mr. Destiny (2010) |
| q24 | a quiet man who loves trains inherits an abandoned little train depot in the cou | The Station Agent | >10 | The Dresser (1983) |
| q25 | single people are sent to a hotel and must find a partner in 45 days or they get | The Lobster | >10 | Chat Room (2002) |
| q26 | อดีตสายลับรัฐบาลออกตามหาลูกสาววัยรุ่นที่โดนแก๊งค้ามนุษย์ลักพาตัวตอนไปเที่ยวยุโรป | Taken | >10 | Missing (2016) |
| q27 | there is a bomb on a train and if it goes below 50 miles per hour it explodes, a | Speed | >10 | The Burning Train (1980) |
| q28 | นักบินอวกาศถูกทิ้งไว้บนดาวอังคารเพราะเพื่อนคิดว่าตายแล้ว ต้องปลูกมันฝรั่งเพื่อเอ | The Martian | >10 | Iceman (1984) |
| q29 | two college friends go looking for their old genius classmate from engineering s | 3 Idiots | >10 | Mujhe Kucch Kehna Hai (2001) |
| q30 | นักเชลโลตกงานกลับไปบ้านเกิด แล้วได้งานใหม่เป็นคนจัดเตรียมศพก่อนใส่โลง ภรรยาไม่ยอ | Departures | >10 | Sanam Teri Kasam (2009) |
| q31 | some movie from the 80s or 90s where a guy keeps living the same day over and ov | Groundhog Day | >10 | Loudspeaker (2009) |
| q38 | หนังที่มีตัวตลก น่ากลัว หลอกเด็กๆ ชื่อเรื่องว่าอะไร | It | >10 | Philips and the Monkey Pen (2013) |
| q39 | Leonardo DiCaprio plays a young con man who pretends to be a pilot and a doctor  | Catch Me If You Can | >10 | Loaded Weapon 1 (1993) |
| q40 | หนังของ Spielberg ยุค 90 ที่นักวิทยาศาสตร์ชุบชีวิตไดโนเสาร์บนเกาะ แล้วไดโนเสาร์ห | Jurassic Park | >10 | Akhil (2015) |
| q41 | the Christopher Nolan movie where a farmer dad leaves his daughter and flies thr | Interstellar | >10 | Hours (2013) |
| q42 | Miyazaki anime where a girl ends up working in a bathhouse for spirits after her | Spirited Away | 2 | The Quiet Room (1996) |
| q43 | late 90s Keanu Reeves movie, he finds out the world is a computer simulation and | The Matrix | >10 | Zegapain ADP (2016) |
| q44 | หนังที่ Jim Carrey กับแฟนเก่าไปจ้างบริษัทลบความทรงจำเกี่ยวกับกันและกัน | Eternal Sunshine of the Spotless Mind | 8 | Barney's Version (2010) |
| q46 | Tom Hanks plays a slow but kind man who keeps ending up in famous moments of Ame | Forrest Gump | >10 | A Modern Twain Story: The Prince and the Pauper (2007) |
| q47 | Ang Lee martial arts film, people fight while flying over bamboo trees and a leg | Crouching Tiger, Hidden Dragon | >10 | Aces Go Places V (1989) |
| q48 | the Mad Max with Mel Gibson where a woman drives a war rig across the desert to  | Mad Max: Fury Road | >10 | The Odd Couple II (1998) |
