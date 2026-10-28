"""A small Okapi BM25 over chunks -- the keyword-only BASELINE we compare against.

Written out (instead of a library) so the baseline is transparent and has no extra dependency.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an and are as at be but by for from has he her his in into is it its of on or she "
    "that the their them they this to was were where which who with".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class BM25Index:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self._tfs = [Counter(tokenize(d)) for d in docs]
        self._lens = np.array([sum(tf.values()) for tf in self._tfs], dtype=np.float32)
        self._avg = float(self._lens.mean()) if len(docs) else 0.0
        df: Counter[str] = Counter()
        for tf in self._tfs:
            df.update(tf.keys())
        n = len(docs)
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        # inverted index: term -> list of (doc_idx, tf)
        self._postings: dict[str, list[tuple[int, int]]] = {}
        for i, tf in enumerate(self._tfs):
            for t, c in tf.items():
                self._postings.setdefault(t, []).append((i, c))

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self._tfs), dtype=np.float32)
        for t in set(tokenize(query)):
            idf = self._idf.get(t)
            if idf is None:
                continue
            for i, tf in self._postings[t]:
                denom = tf + self.k1 * (1 - self.b + self.b * self._lens[i] / self._avg)
                out[i] += idf * tf * (self.k1 + 1) / denom
        return out
