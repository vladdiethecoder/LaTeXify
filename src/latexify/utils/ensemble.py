"""Lightweight utilities for multi-model ensemble voting."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, List, Sequence


@dataclass
class EnsembleCandidate:
    """Single ensemble hypothesis produced by a model."""

    name: str
    payload: Any
    score: float = 0.0
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def weighted_score(self) -> float:
        return self.score * self.weight


class EnsembleVoter:
    """Aggregate multiple model outputs and pick the most confident candidate."""

    def __init__(self, threshold: float = 0.0) -> None:
        self.threshold = threshold
        self._candidates: List[EnsembleCandidate] = []

    def add(
        self,
        name: str,
        payload: Any,
        *,
        score: float,
        weight: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._candidates.append(
            EnsembleCandidate(name=name, payload=payload, score=score, weight=weight, metadata=metadata or {})
        )

    def extend(self, candidates: Iterable[EnsembleCandidate]) -> None:
        for candidate in candidates:
            self._candidates.append(candidate)

    def best_candidate(self) -> EnsembleCandidate | None:
        if not self._candidates:
            return None
        ranked = sorted(self._candidates, key=lambda item: item.weighted_score, reverse=True)
        top = ranked[0]
        if top.weighted_score >= self.threshold:
            return top
        return None

    def best(self, default: Any = None) -> Any:
        candidate = self.best_candidate()
        return candidate.payload if candidate else default

    def consensus(self) -> tuple[Any, float] | None:
        if not self._candidates:
            return None
        ranked = sorted(self._candidates, key=lambda item: item.weighted_score, reverse=True)
        top = ranked[0]
        total = sum(abs(candidate.weighted_score) for candidate in ranked) or 1.0
        confidence = min(1.0, abs(top.weighted_score) / total)
        if top.weighted_score < self.threshold:
            return None
        return top.payload, confidence

    def candidates(self) -> Sequence[EnsembleCandidate]:
        return list(self._candidates)

    def clear(self) -> None:
        self._candidates.clear()

    def __bool__(self) -> bool:
        return bool(self._candidates)


__all__ = ["EnsembleCandidate", "EnsembleVoter"]
