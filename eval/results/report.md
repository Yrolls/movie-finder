# Evaluation report

_Generated 2026-10-28 16:36 UTC — 32 answerable + 6 unanswerable queries from `eval/queries.jsonl`._

LLM: `gemma-4-E4B-it` · embeddings: `intfloat/multilingual-e5-small` · temperature 0

## 1. Retrieval (answerable queries)

| index | chunk words | mode | recall@1 | recall@5 | recall@10 | MRR@10 |
|---|---|---|---|---|---|---|
| `data/index` | 200 | bm25 (baseline) | 31.2% | 46.9% | 46.9% | 0.376 |
| `data/index` | 200 | dense | 15.6% | 31.2% | 34.4% | 0.213 |
| `data/index` | 200 | hybrid | 34.4% | 46.9% | 50.0% | 0.393 |
| `data/index` | 200 | bm25+rewrite | 37.5% | 62.5% | 75.0% | 0.497 |
| `data/index` | 200 | dense+rewrite | 18.8% | 31.2% | 37.5% | 0.242 |
| `data/index` | 200 | hybrid+rewrite | 50.0% | 75.0% | 75.0% | 0.588 |

### recall@5 by query tag (`data/index`)

| tag | n | bm25 | dense | hybrid | bm25+rewrite | dense+rewrite | hybrid+rewrite |
|---|---|---|---|---|---|---|---|
| confusable | 3 | 66.7% | 0.0% | 66.7% | 66.7% | 0.0% | 33.3% |
| en | 19 | 78.9% | 52.6% | 78.9% | 63.2% | 42.1% | 73.7% |
| famous | 24 | 41.7% | 33.3% | 45.8% | 66.7% | 33.3% | 79.2% |
| obscure | 8 | 62.5% | 25.0% | 50.0% | 50.0% | 25.0% | 62.5% |
| partial | 2 | 50.0% | 50.0% | 50.0% | 50.0% | 50.0% | 50.0% |
| plain | 24 | 41.7% | 37.5% | 41.7% | 62.5% | 37.5% | 79.2% |
| th | 13 | 0.0% | 0.0% | 0.0% | 61.5% | 15.4% | 76.9% |
| vague | 2 | 0.0% | 0.0% | 0.0% | 50.0% | 0.0% | 50.0% |
| wrong_detail | 2 | 100.0% | 0.0% | 100.0% | 50.0% | 0.0% | 100.0% |

Rows marked `+rewrite` search the LLM-rewritten query (`eval/results/rewrites.json`). Rewrites that named the gold movie: q14.

## 2. `min_score` floor (dense)

| min_score | recall@5 kept (answerable) | returns nothing (unanswerable) | recall@5 kept, rewritten | returns nothing, rewritten |
|---|---|---|---|---|
| 0.00 | 31.2% | 0.0% | 31.2% | 0.0% |
| 0.76 | 31.2% | 0.0% | 31.2% | 0.0% |
| 0.78 | 31.2% | 0.0% | 31.2% | 0.0% |
| 0.80 | 31.2% | 16.7% | 31.2% | 0.0% |
| 0.82 | 31.2% | 33.3% | 31.2% | 16.7% |
| 0.84 | 15.6% | 83.3% | 15.6% | 50.0% |
| 0.86 | 6.2% | 100.0% | 3.1% | 100.0% |
| 0.88 | 0.0% | 100.0% | 0.0% | 100.0% |

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
