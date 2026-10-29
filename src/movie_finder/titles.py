"""Title normalisation and lookup, shared by the eval (gold labels) and the agent's check_title tool."""
from __future__ import annotations

import difflib
import re
import unicodedata
from collections.abc import Iterable

from .schemas import Movie

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_YEAR = re.compile(r"\(?\b(19\d{2}|20\d{2})\b\)?\s*$")


def norm_title(t: str) -> str:
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode().lower()
    t = _NON_ALNUM.sub(" ", t).strip()
    return re.sub(r"^(the|a|an) ", "", t)


def split_year(text: str) -> tuple[str, int | None]:
    """'Annabelle (2014)' -> ('Annabelle', 2014); 'Annabelle' -> ('Annabelle', None)."""
    m = _YEAR.search(text.strip())
    if m and m.start() > 0:
        return text[: m.start()].strip(" -,"), int(m.group(1))
    return text.strip(), None


class TitleIndex:
    """Exact (normalised) title lookup with a strict fuzzy fallback for small spelling slips."""

    def __init__(self, movies: Iterable[Movie]):
        self._by_norm: dict[str, list[Movie]] = {}
        for m in movies:
            self._by_norm.setdefault(norm_title(m.title), []).append(m)

    def find(self, title: str, year: int | None = None, *, cutoff: float = 0.9) -> list[Movie]:
        key = norm_title(title)
        if not key:
            return []
        found = self._by_norm.get(key)
        if not found:
            close = difflib.get_close_matches(key, self._by_norm.keys(), n=1, cutoff=cutoff)
            found = self._by_norm[close[0]] if close else []
        if year is not None:
            exact = [m for m in found if m.year == year]
            near = [m for m in found if abs(m.year - year) <= 1]
            found = exact or near or found          # people misremember the year by one
        # the corpus lists some films twice (e.g. an Australian and an American row): keep one
        out, seen = [], set()
        for m in sorted(found, key=lambda m: m.movie_id):
            if (norm_title(m.title), m.year) not in seen:
                seen.add((norm_title(m.title), m.year))
                out.append(m)
        return out
