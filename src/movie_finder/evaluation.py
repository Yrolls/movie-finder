"""Evaluation helpers: gold-set loading, title->movie_id resolution, and metrics.

The gold set (eval/queries.jsonl) names the target movie by title + year, not by id, so it
stays valid if the corpus is rebuilt. `"title": null` marks a query whose movie is NOT in the
corpus -- the right behaviour there is to return nothing.
"""
from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .schemas import Movie

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def norm_title(t: str) -> str:
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode().lower()
    t = _NON_ALNUM.sub(" ", t).strip()
    return re.sub(r"^(the|a|an) ", "", t)


@dataclass
class GoldQuery:
    qid: str
    query: str
    title: str | None
    year: int | None
    tags: list[str] = field(default_factory=list)
    movie_id: str | None = None          # filled in by resolve_gold
    also: list[dict] = field(default_factory=list)       # other accepted answers, e.g. a remake
    movie_ids: list[str] = field(default_factory=list)   # all accepted ids (movie_id first)

    @property
    def answerable(self) -> bool:
        return self.title is not None

    @property
    def accepted(self) -> list[str]:
        return self.movie_ids or ([self.movie_id] if self.movie_id else [])


def load_gold(path: Path) -> list[GoldQuery]:
    out = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            d = json.loads(line)
            out.append(GoldQuery(qid=d.get("id", f"q{i:02d}"), query=d["query"], title=d.get("title"),
                                 year=d.get("year"), tags=d.get("tags", []), also=d.get("also", [])))
    return out


def resolve_gold(gold: list[GoldQuery], movies: Sequence[Movie], year_tolerance: int = 1) -> list[str]:
    """Fill movie_id / movie_ids for every answerable query. Returns problems (empty = all good)."""
    index: dict[str, list[Movie]] = defaultdict(list)
    for m in movies:
        index[norm_title(m.title)].append(m)

    def one(qid: str, title: str, year: int | None, problems: list[str]) -> str | None:
        cands = [m for m in index.get(norm_title(title), [])
                 if year is None or abs(m.year - year) <= year_tolerance]
        if len(cands) == 1:
            return cands[0].movie_id
        if not cands:
            problems.append(f"{qid}: '{title}' ({year}) not found in corpus")
            return None
        # loose matching ignores "The"/punctuation; break ties by exact title, then exact year
        same_title = [m for m in cands if m.title.casefold() == title.casefold()]
        for narrowed in (same_title, [m for m in same_title if m.year == year],
                         [m for m in cands if m.year == year]):
            if len(narrowed) == 1:
                return narrowed[0].movie_id
        found = ", ".join(f"{m.movie_id} {m.title} ({m.year}, {m.origin})" for m in cands)
        problems.append(f"{qid}: '{title}' is ambiguous: {found}")
        return None

    problems: list[str] = []
    for g in gold:
        if not g.answerable:
            continue
        g.movie_id = one(g.qid, g.title, g.year, problems)
        extra = [one(g.qid, a["title"], a.get("year"), problems) for a in g.also]
        g.movie_ids = [i for i in [g.movie_id, *extra] if i]
    return problems


def find_titles(movies: Iterable[Movie], text: str, limit: int = 15) -> list[Movie]:
    """Substring search on normalized titles -- used by scripts/find_movie.py."""
    q = norm_title(text)
    return [m for m in movies if q in norm_title(m.title)][:limit]


# ---------------------------------------------------------------- metrics
def rank_of(gold: str | Sequence[str], ranked_ids: Sequence[str]) -> int | None:
    """1-based rank of the first accepted id (one id or a list of ids), or None if absent."""
    accepted = {gold} if isinstance(gold, str) else set(gold)
    for i, mid in enumerate(ranked_ids, 1):
        if mid in accepted:
            return i
    return None


def retrieval_metrics(ranks: Sequence[int | None], ks: Sequence[int] = (1, 5, 10)) -> dict[str, float]:
    """Recall@k and MRR (cut at max k) for answerable queries; `ranks` uses None for a miss."""
    n = len(ranks)
    if n == 0:
        return {}
    kmax = max(ks)
    out = {f"recall@{k}": sum(r is not None and r <= k for r in ranks) / n for k in ks}
    out[f"mrr@{kmax}"] = sum(1 / r for r in ranks if r is not None and r <= kmax) / n
    out[f"ndcg@{kmax}"] = ndcg(ranks, kmax)
    out["median_rank"] = median_rank(ranks, kmax + 1)
    if 5 in ks:
        out["recall@5_ci"] = bootstrap_ci([float(r is not None and r <= 5) for r in ranks])
    return out


def ndcg(ranks: Sequence[int | None], k: int = 10) -> float:
    """nDCG@k with one relevant movie per query: 1/log2(rank+1) if found within k, else 0.
    Unlike recall@k it rewards rank 2 more than rank 9."""
    import math

    return sum(1 / math.log2(r + 1) for r in ranks if r is not None and r <= k) / len(ranks) if ranks else float("nan")


def median_rank(ranks: Sequence[int | None], miss: int = 11) -> float:
    """Median rank of the correct movie, counting a miss (not in top 10) as `miss`."""
    vals = sorted(r if r is not None else miss for r in ranks)
    n = len(vals)
    return float("nan") if n == 0 else (vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2)


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """95% bootstrap confidence interval of the mean (fixed seed -> reproducible).
    With ~30 queries one query is ~3 points, so differences smaller than the CI are noise."""
    import random

    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return means[int(n_boot * alpha / 2)], means[int(n_boot * (1 - alpha / 2)) - 1]


def abstain_rate(result_counts: Sequence[int]) -> float:
    """Share of unanswerable queries for which the system returned nothing (higher = better)."""
    return sum(c == 0 for c in result_counts) / len(result_counts) if result_counts else float("nan")
