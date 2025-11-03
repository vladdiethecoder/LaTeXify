from __future__ import annotations

"""Lightweight critic agent placeholder for deterministic orchestration tests."""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from .specialist_router import SpecialistDecision


def _extract_max_attempts(source: Dict | None) -> Optional[int]:
    if not isinstance(source, dict):
        return None
    candidates: List[Optional[int]] = []
    for key in ("critic", "review", "synthesis"):
        sub = source.get(key)
        if isinstance(sub, dict):
            for field in ("max_attempts", "max_retries"):
                val = sub.get(field)
                if isinstance(val, int) and val > 0:
                    candidates.append(val)
    for field in (
        "critic_max_attempts",
        "review_max_attempts",
        "max_attempts",
        "max_retries",
    ):
        val = source.get(field)
        if isinstance(val, int) and val > 0:
            candidates.append(val)
    for value in candidates:
        if value is not None:
            return value
    return None


@dataclass(frozen=True)
class ReviewResult:
    accepted: bool
    feedback: str = ""


class CriticAgent:
    """Simple synchronous critic wrapper.

    The real implementation will route to a 70B verifier model. For the test
    harness we optimistically accept the first draft to keep the pipeline
    deterministic while still exercising the review loop plumbing.
    """

    def __init__(self, plan: Dict | None = None):
        default_attempts = _extract_max_attempts(plan) or 1
        self._default_attempts = max(1, int(default_attempts))

    def max_attempts(self, task: Dict | None = None) -> int:
        attempt_override = _extract_max_attempts(task)
        if isinstance(attempt_override, int) and attempt_override > 0:
            return attempt_override
        return self._default_attempts

    def review(
        self,
        snippet: str,
        *,
        bundle: Dict,
        decision: SpecialistDecision,
        attempt: int,
        feedback_history: Iterable[str],
    ) -> ReviewResult:
        """Return a placeholder acceptance response."""

        # TODO: integrate real critique backend. For now always accept.
        return ReviewResult(accepted=True, feedback="")
