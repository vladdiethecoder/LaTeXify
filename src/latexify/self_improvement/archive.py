from __future__ import annotations

import random
from typing import List, Optional

from .models import AgentVersion


class Archive:
    """
    Maintains the population of agent versions and performs fitness-proportional selection.
    """

    def __init__(self, versions: Optional[List[AgentVersion]] = None, rng: Optional[random.Random] = None):
        self._versions: List[AgentVersion] = versions or []
        self._rng = rng or random.Random()

    def add(self, version: AgentVersion) -> None:
        self._versions.append(version)

    def all(self) -> List[AgentVersion]:
        return list(self._versions)

    def get_top(self, k: int = 1) -> List[AgentVersion]:
        return sorted(self._versions, key=lambda v: v.metrics.score, reverse=True)[:k]

    def __len__(self) -> int:
        return len(self._versions)

    def select_parent(self, temperature: float = 1.0, min_weight: float = 0.01) -> Optional[AgentVersion]:
        """
        Roulette-wheel selection with temperature smoothing to preserve diversity.
        """
        if not self._versions:
            return None

        weights = []
        for v in self._versions:
            base = max(v.metrics.score, min_weight)
            weight = base ** (1.0 / max(temperature, 1e-6))
            weights.append(weight)

        total = sum(weights)
        if total <= 0:
            return self._rng.choice(self._versions)

        pick = self._rng.uniform(0, total)
        cumulative = 0.0
        for v, w in zip(self._versions, weights):
            cumulative += w
            if pick <= cumulative:
                return v
        return self._versions[-1]

    def maybe_prune(self, limit: Optional[int]) -> None:
        if limit is None:
            return
        if len(self._versions) <= limit:
            return
        # Keep best N, drop rest but retain diversity by keeping one random tail.
        sorted_versions = sorted(self._versions, key=lambda v: v.metrics.score, reverse=True)
        kept = sorted_versions[: max(1, limit - 1)]
        tail = sorted_versions[max(1, limit - 1) :]
        if tail:
            kept.append(self._rng.choice(tail))
        self._versions = kept
