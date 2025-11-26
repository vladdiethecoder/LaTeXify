from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple
import math


def _tokenize(text: str) -> Counter:
    tokens = [t.lower() for t in text.split() if t.strip()]
    return Counter(tokens)


def _cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    common = set(a.keys()) & set(b.keys())
    num = sum(a[t] * b[t] for t in common)
    denom_a = math.sqrt(sum(v * v for v in a.values()))
    denom_b = math.sqrt(sum(v * v for v in b.values()))
    if denom_a == 0 or denom_b == 0:
        return 0.0
    return num / (denom_a * denom_b)


class VectorMemory:
    """
    Minimal semantic memory using bag-of-words cosine similarity.
    """

    def __init__(self):
        self._entries: Dict[str, Counter] = {}
        self._raw: Dict[str, str] = {}

    def add(self, key: str, text: str) -> None:
        self._entries[key] = _tokenize(text)
        self._raw[key] = text

    def query(self, text: str, k: int = 3) -> List[Tuple[str, float]]:
        query_vec = _tokenize(text)
        scores = []
        for key, vec in self._entries.items():
            scores.append((key, _cosine(query_vec, vec)))
        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:k]

    def get_raw(self, key: str) -> str:
        return self._raw.get(key, "")
