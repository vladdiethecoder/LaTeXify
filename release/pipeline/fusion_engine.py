"""Advanced fusion strategies shared by snippet fusion and downstream agents."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from ..core import common


@dataclass
class FusionContext:
    """Lightweight container describing the fusion request."""

    chunk: common.Chunk
    snippet: common.Snippet
    decisions: Sequence[Any]
    validation_data: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FusionResult:
    """Normalized fusion result that downstream code can convert to a snippet."""

    latex: str
    source: str
    confidence: float
    scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseFusion:
    """Common interface for fusion strategies."""

    name: str = "base"

    def fuse(self, context: FusionContext) -> FusionResult | None:
        raise NotImplementedError


class RuleBasedFusion(BaseFusion):
    """Heuristic fusion that favors structure-aware snippets."""

    name = "rule_based"

    def fuse(self, context: FusionContext) -> FusionResult | None:
        region = (context.chunk.metadata or {}).get("region_type", "").lower()
        preferred = None
        for decision in context.decisions:
            scores = getattr(decision, "scores", {}) or {}
            candidate = getattr(decision, "candidate", None)
            if candidate is None:
                continue
            metadata = (candidate.metadata or {}).copy()
            if region in {"equation", "formula"} and scores.get("math", 0) >= 0.5:
                preferred = (candidate, scores)
                break
            if region in {"table", "figure"} and candidate.source.startswith("specialist"):
                preferred = (candidate, scores)
                break
            if preferred is None or scores.get("syntax", 0) > (preferred[1].get("syntax", 0) if preferred else 0):
                preferred = (candidate, scores)
        if not preferred:
            return None
        candidate, scores = preferred
        return FusionResult(
            latex=candidate.latex,
            source=f"{self.name}:{candidate.source}",
            confidence=candidate.confidence,
            scores=scores,
            metadata=candidate.metadata or {},
        )


class ConfidenceWeightedFusion(BaseFusion):
    """Fuse snippets by averaging multiple candidates weighted by confidence."""

    name = "confidence_weighted"

    def fuse(self, context: FusionContext) -> FusionResult | None:
        if not context.decisions:
            return None
        top = sorted(context.decisions, key=lambda d: getattr(d, "weighted_score", 0.0), reverse=True)[:3]
        total_weight = sum(getattr(decision, "weighted_score", 0.0) for decision in top) or 1.0
        pieces = []
        aggregate_scores: Dict[str, float] = {}
        for decision in top:
            weight = getattr(decision, "weighted_score", 0.0) / total_weight
            candidate = decision.candidate
            pieces.append(f"% fuse:{candidate.source}:{weight:.2f}\n{candidate.latex}")
            for key, value in (decision.scores or {}).items():
                aggregate_scores[key] = aggregate_scores.get(key, 0.0) + weight * value
        fused_latex = "\n".join(pieces)
        representative = top[0].candidate
        return FusionResult(
            latex=fused_latex,
            source=f"{self.name}:{representative.source}",
            confidence=max(representative.confidence, 0.5),
            scores=aggregate_scores,
            metadata={"weighted_sources": [decision.candidate.source for decision in top]},
        )


class LLMBasedFusion(BaseFusion):
    """Use a callback (LLM / refiner) to merge snippets into a higher quality block."""

    name = "llm_based"

    def __init__(self, merge_callback: Optional[Callable[[common.Chunk, Sequence[Any]], str]] = None) -> None:
        self._merge_callback = merge_callback

    def fuse(self, context: FusionContext) -> FusionResult | None:
        if not context.decisions:
            return None
        if self._merge_callback is None:
            return self._fallback(context)
        try:
            merged = self._merge_callback(context.chunk, context.decisions)
        except Exception:
            return self._fallback(context)
        top = context.decisions[0]
        return FusionResult(
            latex=merged.strip(),
            source=f"{self.name}:{top.candidate.source}",
            confidence=max(top.candidate.confidence, 0.6),
            scores=top.scores,
            metadata={"llm_merge": True},
        )

    def _fallback(self, context: FusionContext) -> FusionResult | None:
        top = context.decisions[0] if context.decisions else None
        if not top:
            return None
        return FusionResult(
            latex=top.candidate.latex,
            source=f"{self.name}-fallback:{top.candidate.source}",
            confidence=top.candidate.confidence,
            scores=top.scores,
            metadata={"llm_merge": False},
        )


class AdaptiveFusion(BaseFusion):
    """Selects a fusion strategy based on validation feedback and historical scores."""

    name = "adaptive"

    def __init__(
        self,
        validation_history: Optional[Dict[str, Any]] = None,
        rule_fusion: Optional[RuleBasedFusion] = None,
        confidence_fusion: Optional[ConfidenceWeightedFusion] = None,
        llm_fusion: Optional[LLMBasedFusion] = None,
    ) -> None:
        self.validation_history = validation_history or {}
        self.rule_fusion = rule_fusion or RuleBasedFusion()
        self.confidence_fusion = confidence_fusion or ConfidenceWeightedFusion()
        self.llm_fusion = llm_fusion or LLMBasedFusion()

    def fuse(self, context: FusionContext) -> FusionResult | None:
        stats = context.validation_data or self.validation_history
        errors = stats.get("errors") if isinstance(stats, dict) else None
        branch_consistency = 1.0
        try:
            branch_consistency = float(stats.get("branch_consistency", 1.0)) if isinstance(stats, dict) else 1.0
        except Exception:
            branch_consistency = 1.0
        if errors:
            return self.rule_fusion.fuse(context)
        if branch_consistency < 0.6:
            result = self.llm_fusion.fuse(context)
            if result:
                return result
        result = self.confidence_fusion.fuse(context)
        if result:
            return result
        return self.rule_fusion.fuse(context)


__all__ = [
    "FusionContext",
    "FusionResult",
    "BaseFusion",
    "RuleBasedFusion",
    "ConfidenceWeightedFusion",
    "LLMBasedFusion",
    "AdaptiveFusion",
]
