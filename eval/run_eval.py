"""Reproducible evaluation. Writes eval/results/report.md and eval/results/results.json.

    python eval/run_eval.py                                  # retrieval: bm25 vs dense vs hybrid
    python eval/run_eval.py --index data/index data/index_c100 data/index_c400   # chunk sizes
    python eval/run_eval.py --rewrite                        # + same queries after LLM rewriting
    python eval/run_eval.py --rewrite --multi                # + 3 phrasings per query fused with RRF
    python eval/run_eval.py --agent                          # + end-to-end agent vs LLM-only
    python eval/run_eval.py --gold eval/heldout.jsonl --out eval/results_heldout --rewrite --agent

Parts
1. Retrieval   recall@1/5/10 and MRR@10 on answerable queries, per mode and per index
               (BM25 is the named baseline).
2. min_score   sweep of the dense floor: recall@5 kept on answerable queries vs. share of
               unanswerable queries correctly answered with nothing.
3. Rewrite     (--rewrite, calls the LLM once per query, cached in eval/results/rewrites.json)
               every mode again on the LLM-rewritten English plot-style query.
4. Agent       (--agent, calls the LLM) top-1 accuracy of the full app vs. a no-retrieval
               baseline where the model guesses the title from its own memory.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pydantic import BaseModel  # noqa: E402

from movie_finder.config import INDEX_DIR, PROCESSED_DIR, get_settings  # noqa: E402
from movie_finder.data.corpus import load_movies  # noqa: E402
from movie_finder.evaluation import (  # noqa: E402
    GoldQuery,
    abstain_rate,
    load_gold,
    norm_title,
    rank_of,
    resolve_gold,
    retrieval_metrics,
)
from movie_finder.retrieval.loader import load_retriever  # noqa: E402

KS = (1, 5, 10)
# wide range: similarity scales differ per embedding model (e5 ~0.75-0.9, bge ~0.4-0.8)
SWEEP = [0.0, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.74, 0.76, 0.78, 0.80, 0.82, 0.84, 0.86]
RESULTS_DIR = ROOT / "eval" / "results"


# ---------------------------------------------------------------- retrieval
def eval_retrieval(retriever, gold: list[GoldQuery], modes: list[str],
                   rewrites: dict[str, str] | None = None, suffix: str = "",
                   multi: dict[str, list[str]] | None = None, rerank: bool = False) -> dict:
    """`rewrites` maps original query -> text actually searched (None = search the raw query);
    `multi` maps original query -> several phrasings, searched together and fused with RRF."""
    out = {}
    answerable = [g for g in gold if g.answerable]
    for mode in modes:
        ranks, per_tag = [], defaultdict(list)
        misses, per_query = [], []
        for g in answerable:
            if multi:
                hits = retriever.search_many(multi[g.query], k=max(KS), mode=mode, min_score=0.0,
                                             rerank=rerank)
            else:
                text = rewrites[g.query] if rewrites else g.query
                hits = retriever.search(text, k=max(KS), mode=mode, min_score=0.0, rerank=rerank)
            r = rank_of(g.accepted, [h.movie_id for h in hits])
            ranks.append(r)
            for t in g.tags:
                per_tag[t].append(r)
            if r != 1:
                misses.append({"id": g.qid, "query": g.query, "gold": g.title, "rank": r,
                               "top1": f"{hits[0].title} ({hits[0].year})" if hits else None})
            per_query.append({
                "id": g.qid, "query": g.query, "gold": f"{g.title} ({g.year})", "rank": r,
                "searched": (multi[g.query] if multi else (rewrites[g.query] if rewrites else None)),
                "top5": [{"title": h.title, "year": h.year, "score": h.score,
                          "correct": h.movie_id in g.accepted} for h in hits[:5]],
            })
        out[mode + suffix] = {
            "overall": retrieval_metrics(ranks, KS),
            "per_tag": {t: {"n": len(rs), **retrieval_metrics(rs, KS)} for t, rs in sorted(per_tag.items())},
            "misses": misses,
            "per_query": per_query,
        }
    return out


def sweep_min_score(retriever, gold: list[GoldQuery], rewrites: dict[str, str] | None = None,
                    mode: str = "dense") -> list[dict]:
    """Score every query once (no floor), then apply each threshold offline.
    In hybrid mode each hit carries its dense similarity, which is what the floor is applied to."""
    scored = []
    for g in gold:
        text = rewrites[g.query] if rewrites else g.query
        hits = retriever.search(text, k=5, mode=mode, min_score=0.0, rerank=False)
        scored.append((g, [(h.movie_id, h.score) for h in hits]))

    rows = []
    for t in SWEEP:
        ranks, empties = [], []
        for g, hits in scored:
            kept = [mid for mid, s in hits if s >= t]
            if g.answerable:
                ranks.append(rank_of(g.accepted, kept))
            else:
                empties.append(len(kept))
        rows.append({"min_score": t,
                     "recall@5_answerable": retrieval_metrics(ranks, (5,))["recall@5"],
                     "abstain_unanswerable": abstain_rate(empties)})
    return rows


# ---------------------------------------------------------------- query rewriting
REWRITES_PATH = RESULTS_DIR / "rewrites.json"


def get_rewrites(gold: list[GoldQuery], model: str) -> dict[str, str]:
    """LLM rewrites, cached on disk so the numbers are reproducible without re-calling the model."""
    from movie_finder.retrieval.rewrite import rewrite_query

    cache = json.loads(REWRITES_PATH.read_text(encoding="utf-8")) if REWRITES_PATH.exists() else {}
    changed = False
    for g in gold:
        hit = cache.get(g.query)
        if hit is None or hit.get("model") != model:
            cache[g.query] = {"model": model, "plot": rewrite_query(g.query)}
            changed = True
            print(f"  rewrote {g.qid}")
    if changed:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        REWRITES_PATH.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
    return {g.query: cache[g.query]["plot"] for g in gold}


MULTI_PATH = RESULTS_DIR / "rewrites_multi.json"


def get_multi_rewrites(gold: list[GoldQuery], model: str) -> dict[str, list[str]]:
    """Three phrasings per query (see retrieval/rewrite.py), cached like get_rewrites."""
    from movie_finder.retrieval.rewrite import rewrite_queries

    cache = json.loads(MULTI_PATH.read_text(encoding="utf-8")) if MULTI_PATH.exists() else {}
    changed = False
    for g in gold:
        hit = cache.get(g.query)
        if hit is None or hit.get("model") != model:
            cache[g.query] = {"model": model, "plots": rewrite_queries(g.query)}
            changed = True
            print(f"  multi-rewrote {g.qid}")
    if changed:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        MULTI_PATH.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
    return {g.query: cache[g.query]["plots"] for g in gold}


def title_leaks(gold: list[GoldQuery], rewrites: dict[str, str]) -> list[str]:
    """Rewrites that name the gold movie when the user's own query did not (that would make the
    comparison unfair). A title the user already wrote -- "left home alone" -- is not a leak.
    One-word titles ('Up', 'It', 'Speed') are ordinary words, so they are left to manual review."""
    leaks = []
    for g in gold:
        if not g.answerable or len(norm_title(g.title).split()) < 2:
            continue
        t = f" {norm_title(g.title)} "
        if t in f" {norm_title(rewrites[g.query])} " and t not in f" {norm_title(g.query)} ":
            leaks.append(g.qid)
    return leaks


# ---------------------------------------------------------------- agent vs LLM-only
class Guess(BaseModel):
    title: str | None
    year: int | None = None


def llm_only_guess(query: str, movies_by_title: dict) -> str | None:
    from movie_finder.llm import chat_json

    g = chat_json(
        [{"role": "system", "content": "Identify the movie the user describes from your own knowledge. "
                                       'Reply with ONLY JSON: {"title": str|null, "year": int|null}. '
                                       "Use null if you do not know."},
         {"role": "user", "content": query}],
        Guess,
    )
    if not g.title:
        return None
    for m in movies_by_title.get(norm_title(g.title), []):
        if g.year is None or abs(m.year - g.year) <= 1:
            return m.movie_id
    return "__not_in_corpus__"


def eval_agent(retriever, gold: list[GoldQuery], movies, mode: str) -> dict:
    from movie_finder.agent.loop import run_agent

    by_title = defaultdict(list)
    for m in movies:
        by_title[norm_title(m.title)].append(m)

    rows = []
    for g in gold:
        t0 = time.perf_counter()
        res = run_agent(g.query, retriever, mode=mode)
        dt = time.perf_counter() - t0
        top = res.answer.candidates[0].movie_id if res.answer.candidates else None
        base = llm_only_guess(g.query, by_title)
        status = res.answer.status
        if g.answerable:
            agent_ok = top in g.accepted and status != "not_found"
            base_ok = base in g.accepted
            if status == "found":
                outcome = "found, correct" if agent_ok else "found, WRONG"
            elif status == "not_sure":
                outcome = "not sure, correct on top" if agent_ok else "not sure, wrong on top"
            else:
                outcome = "said not found"
        else:
            agent_ok = status == "not_found"
            base_ok = base is None
            outcome = {"not_found": "said not found", "not_sure": "not sure"}.get(status, "found, WRONG")
        cand_ids = [c.movie_id for c in res.answer.candidates]
        rows.append({"id": g.qid, "answerable": g.answerable, "tags": g.tags,
                     "status": status, "outcome": outcome, "agent_top1": top, "agent_ok": agent_ok,
                     "gold_in_candidates": bool(set(cand_ids) & set(g.accepted)),
                     "gold_retrieved": bool(set(res.retrieved) & set(g.accepted)),
                     # found only because the model named it (check_title), not by plot/metadata search
                     "gold_only_by_title_check": bool(set(res.title_checked) & set(g.accepted)),
                     "top_via": (res.evidence.get(top, {}).get("via") if top else None),
                     "used_title_check": any(t.get("tool") == "check_title" for t in res.trace),
                     "llm_only": base, "llm_only_ok": base_ok,
                     "steps": res.steps, "stopped_by": res.stopped_by, "seconds": round(dt, 1),
                     "searches": [t["arguments"] for t in res.trace if t.get("tool") == "search_movies"],
                     "calls": [f"{t['tool']}({t['arguments']})" for t in res.trace if t.get("tool")],
                     "sql": [t["sql"] for t in res.trace if t.get("sql")]})
        print(f"  {g.qid}: agent={'OK ' if agent_ok else 'MISS'} llm-only={'OK ' if base_ok else 'MISS'} "
              f"({res.steps} steps, {dt:.1f}s)")

    def acc(rs, key):
        return sum(r[key] for r in rs) / len(rs) if rs else float("nan")

    ans = [r for r in rows if r["answerable"]]
    una = [r for r in rows if not r["answerable"]]
    found = [r for r in rows if r["status"] == "found"]
    outcomes = {"answerable": defaultdict(int), "unanswerable": defaultdict(int)}
    for r in rows:
        outcomes["answerable" if r["answerable"] else "unanswerable"][r["outcome"]] += 1
    return {
        "summary": {
            # retrieval vs. selection: was the right movie ever retrieved, and did the agent pick it?
            "gold_retrieved_answerable": acc(ans, "gold_retrieved"),
            "gold_in_candidates_answerable": acc(ans, "gold_in_candidates"),
            # when the agent says "found", how often is it right? (wrong-and-confident is the worst error)
            "precision_of_found": (sum(r["agent_ok"] and r["answerable"] for r in found) / len(found)
                                   if found else float("nan")),
            "n_found": len(found),
            "outcomes": {k: dict(v) for k, v in outcomes.items()},
            "stopped_by": dict(Counter(r["stopped_by"] for r in rows)),
            "used_sql": sum(bool(r["sql"]) for r in rows),
            "used_title_check": sum(r.get("used_title_check", False) for r in rows),
            "correct_via_title_check": sum(r["agent_ok"] and r["answerable"] and r.get("top_via") == "title_check"
                                           for r in rows),
            "gold_only_by_title_check_answerable": acc(ans, "gold_only_by_title_check"),
            "per_tag": {t: {"n": len(rs), "agent": acc(rs, "agent_ok"), "llm_only": acc(rs, "llm_only_ok")}
                        for t, rs in sorted(_by_tag(ans).items())},
            "agent_top1_answerable": acc(ans, "agent_ok"),
            "llm_only_top1_answerable": acc(ans, "llm_only_ok"),
            "agent_abstain_unanswerable": acc(una, "agent_ok"),
            "llm_only_abstain_unanswerable": acc(una, "llm_only_ok"),
            "avg_steps": sum(r["steps"] for r in rows) / len(rows),
            "avg_seconds": sum(r["seconds"] for r in rows) / len(rows),
        },
        "rows": rows,
    }


def _by_tag(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        for t in r["tags"]:
            out[t].append(r)
    return out


# ---------------------------------------------------------------- report
def pct(x: float) -> str:
    return "n/a" if x != x else f"{100 * x:.1f}%"


def _embed_models(res: dict) -> str:
    """Model each index was built with (from its meta.json), not the EMBED_MODEL setting."""
    seen = {}
    for r in res.get("retrieval", []):
        seen.setdefault(r["index"], (r.get("meta") or {}).get("embed_model", res["config"]["embed_model"]))
    if not seen:
        return f"`{res['config']['embed_model']}`"
    return ", ".join(f"`{m}` (`{i}`)" for i, m in seen.items())


def write_report(res: dict) -> str:
    L = [f"# Evaluation report", "",
         f"_Generated {res['generated']} — {res['n_answerable']} answerable + "
         f"{res['n_unanswerable']} unanswerable queries from `{res.get('gold', 'eval/queries.jsonl')}`._", "",
         f"LLM: `{res['config']['llm_model']}` · embeddings: {_embed_models(res)} · "
         f"reranker: `{res['config'].get('rerank_model') or 'none'}` · temperature 0", "",
         "## 1. Retrieval (answerable queries)", "",
         "Each query has one correct movie (or a few accepted versions), so recall@k = hit rate@k. "
         "Precision@k is not reported: with one relevant item it is just recall@k / k. "
         "The 95% CI is a bootstrap over queries — with ~30 queries, one query moves recall by ~3 points, "
         "so differences inside the CI are not reliable.", "",
         "| index | chunk words | mode | recall@1 | recall@5 (95% CI) | recall@10 | MRR@10 | nDCG@10 | median rank |",
         "|---|---|---|---|---|---|---|---|---|"]
    for idx in res["retrieval"]:
        for mode, r in idx["modes"].items():
            o = r["overall"]
            lo, hi = o.get("recall@5_ci", (float("nan"), float("nan")))
            mr = o.get("median_rank", float("nan"))
            L.append(f"| `{idx['index']}` | {idx['meta'].get('chunk_words', '?')} | {mode}"
                     f"{' (baseline)' if mode == 'bm25' else ''} | {pct(o['recall@1'])} | "
                     f"{pct(o['recall@5'])} ({pct(lo)}–{pct(hi)}) | {pct(o['recall@10'])} | "
                     f"{o['mrr@10']:.3f} | {o.get('ndcg@10', float('nan')):.3f} | "
                     f"{'>10' if mr > 10 else f'{mr:g}'} |")

    first = res["retrieval"][0]
    tags = sorted({t for m in first["modes"].values() for t in m["per_tag"]})
    if tags:
        L += ["", f"### recall@5 by query tag (`{first['index']}`)", "",
              "| tag | n | " + " | ".join(first["modes"]) + " |",
              "|---|---|" + "---|" * len(first["modes"])]
        for t in tags:
            n = next(m["per_tag"][t]["n"] for m in first["modes"].values() if t in m["per_tag"])
            cells = [pct(m["per_tag"].get(t, {}).get("recall@5", float("nan"))) for m in first["modes"].values()]
            L.append(f"| {t} | {n} | " + " | ".join(cells) + " |")

    if res.get("rewrite_leaks") is not None:
        leaks = res["rewrite_leaks"]
        L += ["", f"Rows marked `+rewrite` search the LLM-rewritten query (`{REWRITES_PATH.relative_to(ROOT).as_posix()}`). "
                  f"Rewrites that named the gold movie: {', '.join(leaks) if leaks else 'none'}."]
    if res.get("multi_leaks") is not None:
        leaks = res["multi_leaks"]
        L += ["", f"Rows marked `+multi` search 3 LLM phrasings per query fused with RRF "
                  f"(`eval/results/rewrites_multi.json`). Phrasings that named the gold movie: "
                  f"{', '.join(leaks) if leaks else 'none'}."]

    if res.get("sweep"):
        cols = [("dense, raw", res["sweep"])]
        if res.get("sweep_rewrite"):
            cols.append(("dense, rewritten", res["sweep_rewrite"]))
        if res.get("sweep_hybrid_rewrite"):
            cols.append(("**hybrid, rewritten (as the app runs)**", res["sweep_hybrid_rewrite"]))
        L += ["", "## 2. `min_score` floor on the dense similarity", "",
              "Each pair: recall@5 kept on answerable queries / share of unanswerable queries "
              "that return nothing.", "",
              "| min_score | " + " | ".join(f"{name}: recall@5 | {name}: nothing" for name, _ in cols) + " |",
              "|---|" + "---|---|" * len(cols)]
        for i, r in enumerate(res["sweep"]):
            cells = []
            for _, rows in cols:
                cells += [pct(rows[i]["recall@5_answerable"]), pct(rows[i]["abstain_unanswerable"])]
            L.append(f"| {r['min_score']:.2f} | " + " | ".join(cells) + " |")

    if res.get("agent"):
        s = res["agent"]["summary"]
        L += ["", f"## 3. End-to-end: agent ({res['config']['retrieval_mode']} retrieval) vs. LLM-only (no retrieval)", "",
              "| system | top-1 accuracy (answerable) | says not found (unanswerable) |",
              "|---|---|---|",
              f"| LLM-only baseline | {pct(s['llm_only_top1_answerable'])} | {pct(s['llm_only_abstain_unanswerable'])} |",
              f"| Agent + retrieval | {pct(s['agent_top1_answerable'])} | {pct(s['agent_abstain_unanswerable'])} |",
              "", f"Average {s['avg_steps']:.1f} model turns and {s['avg_seconds']:.1f}s per query."]
        if "gold_retrieved_answerable" in s:
            L += ["", "### Where the agent loses answers (answerable queries)", "",
                  "| step | share of answerable queries |", "|---|---|",
                  f"| correct movie appeared in any tool result (searches, SQL, check_title) | {pct(s['gold_retrieved_answerable'])} |",
                  f"| correct movie among the agent's ≤3 candidates | {pct(s['gold_in_candidates_answerable'])} |",
                  f"| correct movie ranked first by the agent | {pct(s['agent_top1_answerable'])} |",
                  "", "100% minus the first row is a retrieval error (the right movie was never retrieved); "
                  "the drop from the first row to the last is a selection error (retrieved, but not ranked first).", "",
                  f"Precision of \"found\": when the agent says it found the movie, it is right "
                  f"{pct(s['precision_of_found'])} of the time ({s['n_found']} \"found\" answers).", "",
                  "| outcome | answerable | unanswerable |", "|---|---|---|"]
            oc = s["outcomes"]
            for name in ["found, correct", "not sure, correct on top", "not sure, wrong on top",
                         "found, WRONG", "not sure", "said not found"]:
                a, u = oc["answerable"].get(name, 0), oc["unanswerable"].get(name, 0)
                if a or u:
                    L.append(f"| {name} | {a} | {u} |")
        if "stopped_by" in s:
            sb = ", ".join(f"{k}: {v}" for k, v in sorted(s["stopped_by"].items()))
            L += ["", f"How runs ended — {sb}. Queries where the agent used the metadata database "
                      f"(text-to-SQL): {s['used_sql']}."]
            if "used_title_check" in s:
                L += ["", f"`check_title` (the model names films it knows, the database verifies them): "
                          f"used on {s['used_title_check']} queries; correct answers that came only from it: "
                          f"{s['correct_via_title_check']}; right film reached only through it: "
                          f"{pct(s['gold_only_by_title_check_answerable'])} of answerable queries."]
        if s.get("per_tag"):
            L += ["", "### Top-1 accuracy by query tag (answerable)", "",
                  "| tag | n | agent | LLM-only |", "|---|---|---|---|"]
            for t, v in s["per_tag"].items():
                L.append(f"| {t} | {v['n']} | {pct(v['agent'])} | {pct(v['llm_only'])} |")

    for idx in res["retrieval"][:1]:
        if "dense" in idx["modes"]:
            L += ["", "## Dense misses (rank ≠ 1) — for error analysis", "",
                  "| id | query | gold | rank | dense top-1 |", "|---|---|---|---|---|"]
            for m in idx["modes"]["dense"]["misses"]:
                q = m["query"].replace("|", "/")
                L.append(f"| {m['id']} | {q[:80]} | {m['gold']} | {m['rank'] or '>10'} | {m['top1']} |")
    return "\n".join(L) + "\n"


def write_per_query(res: dict, path: Path) -> None:
    """Top-5 retrieved movies for every query, baseline next to the best configuration."""
    modes = res["retrieval"][0]["modes"]
    show = [m for m in ("bm25", "hybrid+rewrite", "hybrid+multi", "hybrid") if m in modes][:2]
    L = ["# Top-5 retrieved movies per query", "",
         f"_From `{res['retrieval'][0]['index']}`, modes: {', '.join(f'`{m}`' for m in show)}. "
         "✓ marks the correct movie; the number is the dense similarity of its best passage "
         "(BM25 rows show the BM25 score)._", ""]
    for i, q in enumerate(modes[show[0]]["per_query"]):
        L += [f"## {q['id']} — {q['gold']}", "", f"> {q['query']}", ""]
        for m in show:
            pq = modes[m]["per_query"][i]
            rank = pq["rank"] if pq["rank"] is not None else ">10"
            L.append(f"**{m}** — correct movie at rank {rank}")
            if pq.get("searched"):
                searched = pq["searched"] if isinstance(pq["searched"], str) else " / ".join(pq["searched"])
                L.append(f"<br>_searched:_ {searched}")
            L.append("")
            for j, h in enumerate(pq["top5"], 1):
                L.append(f"{j}. {'✓ ' if h['correct'] else ''}{h['title']} ({h['year']}) · {h['score']:.3f}")
            L.append("")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", type=Path, default=ROOT / "eval" / "queries.jsonl")
    ap.add_argument("--out", type=Path, default=Path("eval") / "results",
                    help="results folder (use a separate one for a held-out set)")
    ap.add_argument("--index", type=Path, nargs="+", default=[INDEX_DIR])
    ap.add_argument("--modes", nargs="+", default=["bm25", "dense", "hybrid"])
    ap.add_argument("--rewrite", action="store_true", help="also evaluate LLM-rewritten queries (uses the LLM)")
    ap.add_argument("--multi", action="store_true",
                    help="also evaluate 3 rewritten phrasings per query fused with RRF (uses the LLM)")
    ap.add_argument("--agent", action="store_true", help="also run the end-to-end agent eval (uses the LLM)")
    ap.add_argument("--rerank", metavar="MODEL", default=None,
                    help="cross-encoder for '+rerank' rows and for the agent, "
                         "e.g. cross-encoder/ms-marco-MiniLM-L-6-v2 (default: RERANK_MODEL from .env)")
    ap.add_argument("--no-sweep", action="store_true")
    args = ap.parse_args()

    global RESULTS_DIR, REWRITES_PATH, MULTI_PATH
    RESULTS_DIR = args.out if args.out.is_absolute() else ROOT / args.out
    REWRITES_PATH, MULTI_PATH = RESULTS_DIR / "rewrites.json", RESULTS_DIR / "rewrites_multi.json"

    s = get_settings()
    movies = load_movies(PROCESSED_DIR / "movies.jsonl")
    gold = load_gold(args.gold)
    problems = resolve_gold(gold, movies)
    if problems:
        print("Fix these gold labels first (use scripts/find_movie.py):")
        print("\n".join("  - " + p for p in problems))
        sys.exit(1)
    n_ans = sum(g.answerable for g in gold)
    print(f"[eval] {n_ans} answerable + {len(gold) - n_ans} unanswerable queries")

    res: dict = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "n_answerable": n_ans, "n_unanswerable": len(gold) - n_ans,
        "gold": (args.gold.resolve().relative_to(ROOT).as_posix()
                 if args.gold.resolve().is_relative_to(ROOT) else str(args.gold)),
        "config": {"llm_model": s.llm_model, "embed_model": s.embed_model, "min_score": s.min_score,
                   "retrieval_mode": s.retrieval_mode, "agent_max_steps": s.agent_max_steps,
                   "rerank_model": args.rerank if args.rerank is not None else s.rerank_model},
        "retrieval": [],
    }
    rewrites = None
    if args.rewrite:
        print("[eval] rewriting queries with the LLM (cached) ...")
        rewrites = get_rewrites(gold, s.llm_model)
        res["rewrite_leaks"] = title_leaks(gold, rewrites)

    multi = None
    if args.multi:
        print("[eval] multi-query rewriting with the LLM (cached) ...")
        multi = get_multi_rewrites(gold, s.llm_model)
        joined = {q: " ".join(v) for q, v in multi.items()}
        res["multi_leaks"] = title_leaks(gold, joined)

    first_retriever = None
    for idx in args.index:
        idx = idx if idx.is_absolute() else ROOT / idx
        print(f"[eval] retrieval on {idx.name} ...")
        r = load_retriever(idx, movies=movies, min_score=0.0, rerank_model=args.rerank)
        first_retriever = first_retriever or r
        modes = eval_retrieval(r, gold, args.modes, rerank=False)
        if rewrites:
            modes |= eval_retrieval(r, gold, args.modes, rewrites=rewrites, suffix="+rewrite", rerank=False)
        if multi:
            modes |= eval_retrieval(r, gold, args.modes, multi=multi, suffix="+multi", rerank=False)
        if r.reranker is not None:
            rr_modes = [m for m in args.modes if m != "dense"]
            modes |= eval_retrieval(r, gold, rr_modes, rewrites=rewrites,
                                    suffix=("+rewrite" if rewrites else "") + "+rerank", rerank=True)
        res["retrieval"].append({"index": str(idx.relative_to(ROOT).as_posix()), "meta": r.store.meta,
                                 "modes": modes})

    if not args.no_sweep:
        print("[eval] min_score sweep ...")
        res["sweep"] = sweep_min_score(first_retriever, gold)
        if rewrites:
            res["sweep_rewrite"] = sweep_min_score(first_retriever, gold, rewrites)
            # the floor as the app applies it: hybrid retrieval on the rewritten query
            res["sweep_hybrid_rewrite"] = sweep_min_score(first_retriever, gold, rewrites, mode="hybrid")

    if args.agent:
        print("[eval] agent vs LLM-only (calls the LLM) ...")
        first_retriever.min_score = s.min_score
        res["agent"] = eval_agent(first_retriever, gold, movies, s.retrieval_mode)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    write_per_query(res, RESULTS_DIR / "per_query.md")
    report = write_report(res)
    (RESULTS_DIR / "report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"[eval] wrote {RESULTS_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
