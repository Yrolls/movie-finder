# Evaluation report

_Generated 2026-10-29 07:03 UTC — 42 answerable + 7 unanswerable queries from `eval/queries.jsonl`._

LLM: `gemma-4-E4B-it` · embeddings: `BAAI/bge-small-en-v1.5` (`data/index_bge`), `intfloat/multilingual-e5-small` (`data/index`) · reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2` · temperature 0

## 1. Retrieval (answerable queries)

Each query has one correct movie (or a few accepted versions), so recall@k = hit rate@k. Precision@k is not reported: with one relevant item it is just recall@k / k. The 95% CI is a bootstrap over queries — with ~30 queries, one query moves recall by ~3 points, so differences inside the CI are not reliable.

| index | chunk words | mode | recall@1 | recall@5 (95% CI) | recall@10 | MRR@10 | nDCG@10 | median rank |
|---|---|---|---|---|---|---|---|---|
| `data/index_bge` | 200 | bm25 (baseline) | 31.0% | 42.9% (28.6%–59.5%) | 45.2% | 0.362 | 0.384 | >10 |
| `data/index_bge` | 200 | dense | 23.8% | 28.6% (14.3%–42.9%) | 31.0% | 0.259 | 0.271 | >10 |
| `data/index_bge` | 200 | hybrid | 38.1% | 42.9% (28.6%–59.5%) | 47.6% | 0.411 | 0.426 | >10 |
| `data/index_bge` | 200 | bm25+rewrite | 35.7% | 61.9% (47.6%–76.2%) | 71.4% | 0.482 | 0.538 | 2 |
| `data/index_bge` | 200 | dense+rewrite | 21.4% | 38.1% (23.8%–52.4%) | 42.9% | 0.293 | 0.326 | >10 |
| `data/index_bge` | 200 | hybrid+rewrite | 50.0% | 73.8% (59.5%–85.7%) | 88.1% | 0.614 | 0.677 | 1.5 |
| `data/index_bge` | 200 | bm25+rewrite+rerank | 54.8% | 78.6% (64.3%–90.5%) | 81.0% | 0.657 | 0.695 | 1 |
| `data/index_bge` | 200 | hybrid+rewrite+rerank | 59.5% | 81.0% (69.0%–92.9%) | 88.1% | 0.699 | 0.743 | 1 |
| `data/index` | 200 | bm25 (baseline) | 31.0% | 42.9% (28.6%–59.5%) | 45.2% | 0.362 | 0.384 | >10 |
| `data/index` | 200 | dense | 14.3% | 28.6% (16.7%–42.9%) | 33.3% | 0.201 | 0.233 | >10 |
| `data/index` | 200 | hybrid | 33.3% | 50.0% (35.7%–64.3%) | 52.4% | 0.393 | 0.424 | 7 |
| `data/index` | 200 | bm25+rewrite | 35.7% | 61.9% (47.6%–76.2%) | 71.4% | 0.482 | 0.538 | 2 |
| `data/index` | 200 | dense+rewrite | 14.3% | 28.6% (16.7%–42.9%) | 40.5% | 0.211 | 0.256 | >10 |
| `data/index` | 200 | hybrid+rewrite | 50.0% | 76.2% (64.3%–88.1%) | 78.6% | 0.596 | 0.642 | 1.5 |
| `data/index` | 200 | bm25+rewrite+rerank | 54.8% | 78.6% (64.3%–90.5%) | 81.0% | 0.657 | 0.695 | 1 |
| `data/index` | 200 | hybrid+rewrite+rerank | 52.4% | 76.2% (61.9%–88.1%) | 81.0% | 0.641 | 0.683 | 1 |

### recall@5 by query tag (`data/index_bge`)

| tag | n | bm25 | dense | hybrid | bm25+rewrite | dense+rewrite | hybrid+rewrite | bm25+rewrite+rerank | hybrid+rewrite+rerank |
|---|---|---|---|---|---|---|---|---|---|
| confusable | 3 | 66.7% | 0.0% | 33.3% | 66.7% | 33.3% | 66.7% | 66.7% | 66.7% |
| en | 27 | 66.7% | 44.4% | 66.7% | 63.0% | 44.4% | 74.1% | 77.8% | 81.5% |
| famous | 34 | 38.2% | 26.5% | 44.1% | 64.7% | 38.2% | 79.4% | 76.5% | 79.4% |
| metadata | 10 | 30.0% | 40.0% | 40.0% | 60.0% | 30.0% | 60.0% | 70.0% | 70.0% |
| obscure | 8 | 62.5% | 37.5% | 37.5% | 50.0% | 37.5% | 50.0% | 87.5% | 87.5% |
| partial | 2 | 50.0% | 0.0% | 50.0% | 50.0% | 0.0% | 50.0% | 50.0% | 50.0% |
| plain | 24 | 41.7% | 33.3% | 41.7% | 62.5% | 50.0% | 79.2% | 83.3% | 91.7% |
| th | 15 | 0.0% | 0.0% | 0.0% | 60.0% | 26.7% | 73.3% | 80.0% | 80.0% |
| vague | 3 | 33.3% | 33.3% | 33.3% | 66.7% | 33.3% | 100.0% | 66.7% | 66.7% |
| wrong_detail | 3 | 100.0% | 33.3% | 100.0% | 66.7% | 33.3% | 100.0% | 100.0% | 66.7% |

Rows marked `+rewrite` search the LLM-rewritten query (`eval/results/rewrites.json`). Rewrites that named the gold movie: none.

## 2. `min_score` floor on the dense similarity

Each pair: recall@5 kept on answerable queries / share of unanswerable queries that return nothing.

| min_score | dense, raw: recall@5 | dense, raw: nothing | dense, rewritten: recall@5 | dense, rewritten: nothing | **hybrid, rewritten (as the app runs)**: recall@5 | **hybrid, rewritten (as the app runs)**: nothing |
|---|---|---|---|---|---|---|
| 0.00 | 28.6% | 0.0% | 38.1% | 0.0% | 73.8% | 0.0% |
| 0.40 | 28.6% | 0.0% | 38.1% | 0.0% | 73.8% | 0.0% |
| 0.50 | 28.6% | 0.0% | 38.1% | 0.0% | 73.8% | 0.0% |
| 0.55 | 28.6% | 0.0% | 35.7% | 0.0% | 71.4% | 0.0% |
| 0.60 | 28.6% | 0.0% | 35.7% | 0.0% | 59.5% | 0.0% |
| 0.65 | 21.4% | 42.9% | 23.8% | 42.9% | 26.2% | 42.9% |
| 0.70 | 4.8% | 85.7% | 9.5% | 85.7% | 9.5% | 85.7% |
| 0.74 | 0.0% | 100.0% | 0.0% | 85.7% | 0.0% | 85.7% |
| 0.76 | 0.0% | 100.0% | 0.0% | 100.0% | 0.0% | 100.0% |
| 0.78 | 0.0% | 100.0% | 0.0% | 100.0% | 0.0% | 100.0% |
| 0.80 | 0.0% | 100.0% | 0.0% | 100.0% | 0.0% | 100.0% |
| 0.82 | 0.0% | 100.0% | 0.0% | 100.0% | 0.0% | 100.0% |
| 0.84 | 0.0% | 100.0% | 0.0% | 100.0% | 0.0% | 100.0% |
| 0.86 | 0.0% | 100.0% | 0.0% | 100.0% | 0.0% | 100.0% |

## Dense misses (rank ≠ 1) — for error analysis

| id | query | gold | rank | dense top-1 |
|---|---|---|---|---|
| q01 | ผู้ชายคนหนึ่งตื่นขึ้นมาแล้วเจอวันเดิมซ้ำไปเรื่อย ๆ ในเมืองเล็ก ๆ ช่วงหน้าหนาว เข | Groundhog Day | >10 | Sworn Brothers (1987) |
| q02 | a team of thieves breaks into a rich heir's mind while he sleeps to plant an ide | Inception | >10 | Acid Factory (2009) |
| q03 | guy can't make new memories so he tattoos notes on his body and uses polaroid ph | Memento | 4 | I.C.U. (2009) |
| q04 | ชายคนหนึ่งค่อย ๆ รู้ว่าชีวิตทั้งชีวิตของเขาเป็นรายการทีวี ทุกคนรอบตัวเป็นนักแสดง | The Truman Show | >10 | Sworn Brothers (1987) |
| q05 | the one where the guy is stuck on an island after a plane crash and his only fri | Cast Away | >10 | Hot Pursuit (1987) |
| q06 | วัยรุ่นย้อนเวลากลับไปด้วยรถสปอร์ต แล้วเผลอทำให้พ่อแม่ตัวเองไม่ได้รักกัน ต้องแก้ใ | Back to the Future | >10 | Sworn Brothers (1987) |
| q07 | a child psychologist tries to help a scared little boy who says he can see ghost | The Sixth Sense | >10 | Clownhouse (1988) |
| q08 | เด็กผู้ชายขอพรกับตู้เกมแล้วตื่นมาตัวโตเป็นผู้ใหญ่ ได้ไปทำงานบริษัทของเล่น | Big | >10 | Sworn Brothers (1987) |
| q10 | a man from eastern europe gets stuck living inside a new york airport for months | The Terminal | 2 | Moonlighting (1982) |
| q11 | คุณลุงแก่ ๆ ผูกลูกโป่งเป็นพันลูกกับบ้านแล้วลอยไปอเมริกาใต้ มีเด็กลูกเสือติดไปด้ว | Up | >10 | Sworn Brothers (1987) |
| q12 | หนูที่อยากเป็นเชฟ แอบควบคุมเด็กหนุ่มในครัวร้านอาหารที่ปารีสด้วยการดึงผมใต้หมวก | Ratatouille | >10 | Sworn Brothers (1987) |
| q13 | kids find an old board game and every roll lets jungle animals and a hunter out  | Jumanji | >10 | Thutturi (2006) |
| q14 | a boy is accidentally left home alone during easter holiday and fights off two b | Home Alone | >10 | Hari Puttar: A Comedy of Terrors (2008) |
| q15 | soldier fighting aliens keeps dying and waking up on the same day before the bat | Edge of Tomorrow | >10 | Alien Origin (2012) |
| q16 | a man wakes up in someone else's body on a commuter train and keeps reliving the | Source Code | >10 | Last Passenger (2013) |
| q17 | ผู้ชายเหงา ๆ ที่เพิ่งหย่า ตกหลุมรักระบบปฏิบัติการ AI ในคอมพิวเตอร์ที่มีแค่เสียงผ | Her | >10 | The Inerasable (2016) |
| q20 | นักวิทยาศาสตร์ผู้หญิงที่ฟังสัญญาณจากอวกาศ ได้รับข้อความจากดาวดวงอื่นที่มีแบบแปลน | Contact | >10 | Sworn Brothers (1987) |
| q23 | ผู้ชายขี้อายพาตุ๊กตาขนาดเท่าคนจริงมาแนะนำว่าเป็นแฟน แล้วคนทั้งเมืองก็เล่นตามไปด้ | Lars and the Real Girl | >10 | Sworn Brothers (1987) |
| q24 | a quiet man who loves trains inherits an abandoned little train depot in the cou | The Station Agent | >10 | Lion (2016) |
| q26 | อดีตสายลับรัฐบาลออกตามหาลูกสาววัยรุ่นที่โดนแก๊งค้ามนุษย์ลักพาตัวตอนไปเที่ยวยุโรป | Taken | >10 | Sworn Brothers (1987) |
| q27 | there is a bomb on a train and if it goes below 50 miles per hour it explodes, a | Speed | 8 | Money Train (1995) |
| q28 | นักบินอวกาศถูกทิ้งไว้บนดาวอังคารเพราะเพื่อนคิดว่าตายแล้ว ต้องปลูกมันฝรั่งเพื่อเอ | The Martian | >10 | Sworn Brothers (1987) |
| q29 | two college friends go looking for their old genius classmate from engineering s | 3 Idiots | >10 | Road Trip (2000) |
| q30 | นักเชลโลตกงานกลับไปบ้านเกิด แล้วได้งานใหม่เป็นคนจัดเตรียมศพก่อนใส่โลง ภรรยาไม่ยอ | Departures | >10 | Sworn Brothers (1987) |
| q31 | some movie from the 80s or 90s where a guy keeps living the same day over and ov | Groundhog Day | >10 | The Reunion (2012) |
| q38 | หนังที่มีตัวตลก น่ากลัว หลอกเด็กๆ ชื่อเรื่องว่าอะไร | It | >10 | Anjali (1990) |
| q39 | Leonardo DiCaprio plays a young con man who pretends to be a pilot and a doctor  | Catch Me If You Can | >10 | The Rise and Fall of a White Collar Hooligan (2012) |
| q40 | หนังของ Spielberg ยุค 90 ที่นักวิทยาศาสตร์ชุบชีวิตไดโนเสาร์บนเกาะ แล้วไดโนเสาร์ห | Jurassic Park | >10 | 9 (2009) |
| q43 | late 90s Keanu Reeves movie, he finds out the world is a computer simulation and | The Matrix | >10 | Chain Reaction (1996) |
| q44 | หนังที่ Jim Carrey กับแฟนเก่าไปจ้างบริษัทลบความทรงจำเกี่ยวกับกันและกัน | Eternal Sunshine of the Spotless Mind | >10 | Fun with Dick and Jane (2005) |
| q46 | Tom Hanks plays a slow but kind man who keeps ending up in famous moments of Ame | Forrest Gump | >10 | Hank Williams First Nation (2004) |
| q47 | Ang Lee martial arts film, people fight while flying over bamboo trees and a leg | Crouching Tiger, Hidden Dragon | >10 | Once Upon a Time in China and America (1997) |
