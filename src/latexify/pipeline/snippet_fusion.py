"""Snippet fusion utilities that combine multiple candidates per chunk."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from statistics import mean
from typing import Dict, List, Sequence

from ..core import common
from .consistency_utils import MathConsistencyValidator
from .fusion_engine import (
    AdaptiveFusion,
    FusionContext,
    FusionResult,
    LLMBasedFusion,
    RuleBasedFusion,
    ConfidenceWeightedFusion,
)
from . import branch_c_fusion


class FusionStrategy(str, Enum):
    SELECT_BEST = "select_best"
    MERGE_HYBRID = "merge_hybrid"
    ENSEMBLE_AVERAGE = "ensemble_average"
    ADAPTIVE = "adaptive"
    MULTI_BRANCH = "multi_branch"


@dataclass
class SnippetCandidate:
    latex: str
    source: str
    confidence: float
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class SnippetDecision:
    candidate: SnippetCandidate
    scores: Dict[str, float]
    weighted_score: float


class SnippetJudge:
    """Score snippet candidates using math, syntax, and style heuristics."""

    def __init__(self, weights: Dict[str, float] | None = None) -> None:
        self.validator = MathConsistencyValidator()
        self.weights = weights or {
            "math": 0.35,
            "syntax": 0.25,
            "style": 0.2,
            "source": 0.2,
        }

    def evaluate(self, chunk: common.Chunk, candidates: Sequence[SnippetCandidate]) -> List[SnippetDecision]:
        decisions: List[SnippetDecision] = []
        for candidate in candidates:
            scores = self.score_candidate(chunk, candidate)
            weighted = sum(self.weights.get(key, 0.0) * scores.get(key, 0.0) for key in self.weights)
            decisions.append(SnippetDecision(candidate=candidate, scores=scores, weighted_score=round(weighted, 3)))
        decisions.sort(key=lambda entry: entry.weighted_score, reverse=True)
        return decisions

    def score_candidate(self, chunk: common.Chunk, candidate: SnippetCandidate) -> Dict[str, float]:
        return self._score_candidate(chunk, candidate)

    def fuse(
        self,
        chunk: common.Chunk,
        decisions: Sequence[SnippetDecision],
        strategy: FusionStrategy,
    ) -> SnippetDecision | None:
        if not decisions:
            return None
        if strategy == FusionStrategy.SELECT_BEST or len(decisions) == 1:
            return decisions[0]
        if strategy == FusionStrategy.MERGE_HYBRID and len(decisions) >= 2:
            merged = self._merge(decisions[0].candidate, decisions[1].candidate)
            merged_candidate = SnippetCandidate(
                latex=merged,
                source=f"{decisions[0].candidate.source}+{decisions[1].candidate.source}",
                confidence=mean([decisions[0].weighted_score, decisions[1].weighted_score]),
                metadata={"merged": True},
            )
            scores = self._score_candidate(chunk, merged_candidate)
            weighted = sum(self.weights.get(key, 0.0) * scores.get(key, 0.0) for key in self.weights)
            return SnippetDecision(merged_candidate, scores, round(weighted, 3))
        if strategy == FusionStrategy.ENSEMBLE_AVERAGE:
            aggregate = decisions[0].candidate.latex
            tail = [d.candidate.latex for d in decisions[1:3] if d.candidate.latex.strip()]
            if tail:
                aggregate += "\n% ensemble-complement\n" + "\n".join(tail)
            synthetic = SnippetCandidate(
                latex=aggregate,
                source="ensemble",
                confidence=mean(dec.weighted_score for dec in decisions[:3]),
                metadata={"ensemble": True},
            )
            scores = self._score_candidate(chunk, synthetic)
            weighted = sum(self.weights.get(key, 0.0) * scores.get(key, 0.0) for key in self.weights)
            return SnippetDecision(synthetic, scores, round(weighted, 3))
        return decisions[0]

    def _merge(self, primary: SnippetCandidate, secondary: SnippetCandidate) -> str:
        if not secondary.latex.strip():
            return primary.latex
        if secondary.source == primary.source:
            return primary.latex
        return f"{primary.latex}\n% hybrid-from:{secondary.source}\n{secondary.latex}"

    def _score_candidate(self, chunk: common.Chunk, candidate: SnippetCandidate) -> Dict[str, float]:
        math_stats = self.validator.validate(chunk.text, candidate.latex)
        math_score = mean(math_stats.values()) if math_stats else 0.0
        syntax_score = self._syntax_score(candidate.latex)
        style_score = self._style_score(candidate.latex)
        return {
            "math": round(math_score, 3),
            "syntax": round(syntax_score, 3),
            "style": round(style_score, 3),
            "source": _clamp(candidate.confidence),
        }

    def _syntax_score(self, latex: str) -> float:
        depth = 0
        imbalance = 0
        for char in latex:
            if char == "{":
                depth += 1
            elif char == "}":
                if depth == 0:
                    imbalance += 1
                else:
                    depth -= 1
        imbalance += depth
        return max(0.0, 1.0 - min(1.0, imbalance / 6))

    def _style_score(self, latex: str) -> float:
        score = 0.0
        for token in ("\\toprule", "\\midrule", "\\begin{align", "\\label", "\\caption"):
            if token in latex:
                score += 0.2
        if "\\begin{figure" in latex:
            score += 0.2
        if "\\begin{table" in latex:
            score += 0.2
        return _clamp(score)


def _plain_text_candidate(text: str) -> str:
    escaped = text.replace("\\", r"\textbackslash{}")
    for char in "&%$#_{}":
        escaped = escaped.replace(char, rf"\{char}")
    return escaped.strip()


def run_snippet_fusion(
    chunks_path: Path,
    snippets_path: Path,
    metrics_path: Path | None,
    validation_path: Path | None,
    output_path: Path,
    *,
    strategy: FusionStrategy = FusionStrategy.SELECT_BEST,
    branch_manifest_path: Path | None = None,
) -> Dict[str, object]:
    chunks = {chunk.chunk_id: chunk for chunk in common.load_chunks(chunks_path)}
    snippets = common.load_snippets(snippets_path)
    metrics_data = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path and metrics_path.exists() else {}
    validation_data = json.loads(validation_path.read_text(encoding="utf-8")) if validation_path and validation_path.exists() else {}
    branch_manifest = branch_c_fusion.load_branch_manifest(branch_manifest_path)
    multi_branch_engine = branch_c_fusion.MultiBranchFusionEngine(branch_manifest)
    judge = SnippetJudge()
    decisions_payload: List[Dict[str, object]] = []
    aggregate_scores: List[float] = []
    flagged: List[str] = []
    rule_engine = RuleBasedFusion()
    confidence_engine = ConfidenceWeightedFusion()
    llm_engine = LLMBasedFusion()
    adaptive_engine = AdaptiveFusion(
        validation_history=validation_data,
        rule_fusion=rule_engine,
        confidence_fusion=confidence_engine,
        llm_fusion=llm_engine,
    )
    for snippet in snippets:
        chunk = chunks.get(snippet.chunk_id)
        if not chunk:
            continue
        branch_context = branch_c_fusion.build_branch_context(chunk, snippet, branch_manifest)
        candidates = _build_candidates(chunk, snippet, branch_context)
        decisions = judge.evaluate(chunk, candidates)
        context = FusionContext(
            chunk=chunk,
            snippet=snippet,
            decisions=decisions,
            validation_data=validation_data,
            metadata={"branch_info": branch_context},
        )
        fusion_result = _run_fusion_strategy(
            strategy,
            context,
            rule_engine=rule_engine,
            confidence_engine=confidence_engine,
            llm_engine=llm_engine,
            adaptive_engine=adaptive_engine,
            multi_branch_engine=multi_branch_engine,
        )
        fused = _fusion_result_to_decision(judge, chunk, fusion_result)
        if fused is None:
            continue
        aggregate_scores.append(fused.weighted_score)
        if fused.weighted_score < 0.45:
            flagged.append(snippet.chunk_id)
        decisions_payload.append(
            {
                "chunk_id": snippet.chunk_id,
                "selected_source": fused.candidate.source,
                "score": fused.weighted_score,
                "scores": fused.scores,
                "candidates": [
                    {
                        "source": decision.candidate.source,
                        "score": decision.weighted_score,
                        "confidence": decision.candidate.confidence,
                    }
                    for decision in decisions
                ],
            }
        )
    payload = {
        "strategy": strategy.value,
        "decisions": decisions_payload,
        "aggregate_confidence": round(mean(aggregate_scores), 3) if aggregate_scores else 0.0,
        "flagged_chunks": flagged,
        "compile_success": bool(validation_data.get("success", True)),
        "compile_errors": validation_data.get("errors", []),
        "metrics_snapshot": {
            "section_fidelity": metrics_data.get("section_fidelity"),
            "equation_fidelity": metrics_data.get("equation_fidelity"),
        },
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _build_candidates(
    chunk: common.Chunk,
    snippet: common.Snippet,
    branch_context: Dict[str, object] | None,
) -> List[SnippetCandidate]:
    notes = snippet.notes or {}
    base_conf = float(notes.get("snippet_confidence", 0.55))
    source = notes.get("snippet_source", "specialist")
    candidates = [
        SnippetCandidate(latex=snippet.latex, source=source, confidence=base_conf, metadata=dict(notes)),
    ]
    branch_c_fusion.annotate_candidate(
        candidates[0],
        (branch_context or {}).get("primary_branch", branch_c_fusion.BRANCH_C),
    )
    if chunk.text and chunk.text.strip():
        fallback = _plain_text_candidate(chunk.text)
        if fallback and fallback not in snippet.latex:
            fallback_candidate = SnippetCandidate(
                latex=f"\\par {fallback} \\",
                source="chunk-text",
                confidence=0.2,
                metadata={"fallback": True},
            )
            branch_c_fusion.annotate_candidate(fallback_candidate, branch_c_fusion.BRANCH_A)
            candidates.append(fallback_candidate)
    branch_payloads = branch_c_fusion.extra_branch_candidates(chunk, snippet, branch_context or {})
    for payload in branch_payloads:
        candidates.append(SnippetCandidate(**payload))
    return candidates


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _run_fusion_strategy(
    strategy: FusionStrategy,
    context: FusionContext,
    *,
    rule_engine: RuleBasedFusion,
    confidence_engine: ConfidenceWeightedFusion,
    llm_engine: LLMBasedFusion,
    adaptive_engine: AdaptiveFusion,
    multi_branch_engine: branch_c_fusion.MultiBranchFusionEngine,
) -> FusionResult | None:
    if strategy == FusionStrategy.SELECT_BEST:
        return rule_engine.fuse(context)
    if strategy == FusionStrategy.MERGE_HYBRID:
        return confidence_engine.fuse(context)
    if strategy == FusionStrategy.ENSEMBLE_AVERAGE:
        return llm_engine.fuse(context)
    if strategy == FusionStrategy.ADAPTIVE:
        return adaptive_engine.fuse(context)
    if strategy == FusionStrategy.MULTI_BRANCH:
        return multi_branch_engine.fuse(context)
    return rule_engine.fuse(context)


def _fusion_result_to_decision(
    judge: SnippetJudge,
    chunk: common.Chunk,
    result: FusionResult | None,
) -> SnippetDecision | None:
    if result is None or not result.latex:
        return None
    candidate = SnippetCandidate(
        latex=result.latex,
        source=result.source,
        confidence=_clamp(result.confidence),
        metadata=result.metadata,
    )
    scores = result.scores or judge.score_candidate(chunk, candidate)
    weighted = sum(judge.weights.get(key, 0.0) * scores.get(key, 0.0) for key in judge.weights)
    return SnippetDecision(candidate=candidate, scores=scores, weighted_score=round(weighted, 3))


__all__ = ["FusionStrategy", "SnippetJudge", "run_snippet_fusion"]
