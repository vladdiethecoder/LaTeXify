"""Multi-branch fusion helpers for Branch C."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, MutableMapping, Sequence

from ..core import common
from .fusion_engine import BaseFusion, FusionContext, FusionResult, ConfidenceWeightedFusion, RuleBasedFusion

if TYPE_CHECKING:
    from .snippet_fusion import SnippetCandidate, SnippetDecision
else:  # pragma: no cover - runtime fallback to avoid circular import
    SnippetCandidate = Any
    SnippetDecision = Any

DEFAULT_CONFIDENCE = 0.75
BRANCH_A = "branch_a"
BRANCH_B = "branch_b"
BRANCH_C = "branch_c"


def load_branch_manifest(path: Path | None) -> Dict[str, object]:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def branch_metrics_map(manifest: Dict[str, object]) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    results = manifest.get("results") if isinstance(manifest, dict) else None
    if isinstance(results, list):
        for entry in results:
            branch_name = entry.get("branch") if isinstance(entry, MutableMapping) else None
            if not branch_name:
                continue
            entry_metrics = entry.get("metrics") if isinstance(entry, MutableMapping) else None
            confidence = DEFAULT_CONFIDENCE
            if isinstance(entry_metrics, MutableMapping):
                for key in ("avg_confidence", "confidence", "coverage"):
                    value = entry_metrics.get(key)
                    if isinstance(value, (int, float)):
                        confidence = max(0.1, float(value))
                        break
            metrics[branch_name] = {
                "confidence": confidence,
                "failures": float(entry_metrics.get("failures", 0.0)) if isinstance(entry_metrics, MutableMapping) else 0.0,
            }
    return metrics


def determine_primary_branch(snippet: common.Snippet) -> str:
    notes = snippet.notes or {}
    branch_meta = notes.get("branch")
    if isinstance(branch_meta, dict):
        return BRANCH_B
    return BRANCH_C


def build_branch_context(
    chunk: common.Chunk,
    snippet: common.Snippet,
    manifest: Dict[str, object],
) -> Dict[str, object]:
    context = {
        "primary_branch": determine_primary_branch(snippet),
        "chunk_region": (chunk.metadata or {}).get("region_type"),
        "manifest": manifest,
        "branch_metrics": branch_metrics_map(manifest),
    }
    return context


def annotate_candidate(candidate: SnippetCandidate, branch_label: str) -> None:
    metadata = dict(candidate.metadata or {})
    metadata["branch"] = branch_label
    candidate.metadata = metadata


def extra_branch_candidates(
    chunk: common.Chunk,
    snippet: common.Snippet,
    manifest: Dict[str, object],
) -> List[Dict[str, object]]:
    payloads: List[Dict[str, object]] = []
    notes = snippet.notes or {}
    branch_meta = notes.get("branch")
    if isinstance(branch_meta, dict):
        branch_label = branch_meta.get("branch_id") or branch_meta.get("region_type") or BRANCH_B
        figure_hint = (branch_meta.get("region_type") or "").lower()
        vision_blurb = branch_meta.get("extras", {}).get("caption") if isinstance(branch_meta.get("extras"), dict) else None
        if vision_blurb:
            latex = f"% branch:{branch_label}\n{vision_blurb}"
            payloads.append(
                {
                    "latex": latex,
                    "source": "branch_b_vision",
                    "confidence": 0.5,
                    "metadata": {"branch": BRANCH_B, "region": figure_hint},
                }
            )
    return payloads


class MultiBranchFusionEngine(BaseFusion):
    name = "multi_branch"

    def __init__(self, manifest: Dict[str, object] | None = None) -> None:
        self.manifest = manifest or {}
        self.branch_metrics = branch_metrics_map(self.manifest)
        self.confidence_fusion = ConfidenceWeightedFusion()
        self.rule_fusion = RuleBasedFusion()

    def fuse(self, context: FusionContext) -> FusionResult | None:
        if not context.decisions:
            return None
        region = (context.chunk.metadata or {}).get("region_type", "").lower()
        branch_groups = self._group_by_branch(context.decisions)
        if region in {"equation", "formula"} and BRANCH_B in branch_groups:
            return self._prefer_branch(branch_groups[BRANCH_B], BRANCH_B)
        if region in {"text", "paragraph"} and BRANCH_A in branch_groups:
            best = branch_groups[BRANCH_A][0]
            if best.weighted_score >= 0.55:
                return self._prefer_branch(branch_groups[BRANCH_A], BRANCH_A)
        weighted_segments: List[str] = []
        aggregate_scores: Dict[str, float] = {}
        provenance: List[Dict[str, object]] = []
        weighted = self._weight_decisions(context.decisions)
        if not weighted:
            return self.confidence_fusion.fuse(context)
        total = sum(weight for weight, _ in weighted)
        for weight, decision in weighted[:3]:
            candidate = decision.candidate
            branch = candidate.metadata.get("branch", BRANCH_C)
            normalized = weight / total if total else 0.0
            provenance.append({"branch": branch, "weight": round(normalized, 3), "source": candidate.source})
            segment = f"% branch:{branch} weight={normalized:.2f}\n{candidate.latex}"
            weighted_segments.append(segment)
            for key, value in (decision.scores or {}).items():
                aggregate_scores[key] = aggregate_scores.get(key, 0.0) + normalized * value
        fused_text = "\n".join(weighted_segments)
        top_candidate = weighted[0][1].candidate
        metadata = {
            "provenance": provenance,
            "branch_metrics": self.branch_metrics,
            "validation": context.metadata.get("branch_info"),
        }
        return FusionResult(
            latex=fused_text,
            source=f"{self.name}:{top_candidate.source}",
            confidence=max(top_candidate.confidence, 0.6),
            scores=aggregate_scores,
            metadata=metadata,
        )

    def _prefer_branch(self, decisions: List[SnippetDecision], branch: str) -> FusionResult | None:
        if not decisions:
            return None
        top = decisions[0]
        candidate = top.candidate
        return FusionResult(
            latex=candidate.latex,
            source=f"{self.name}:{candidate.source}",
            confidence=max(candidate.confidence, 0.55),
            scores=top.scores,
            metadata={"provenance": [{"branch": branch, "weight": 1.0, "source": candidate.source}]},
        )

    def _group_by_branch(self, decisions: Sequence[SnippetDecision]) -> Dict[str, List[SnippetDecision]]:
        groups: Dict[str, List[SnippetDecision]] = {}
        for decision in decisions:
            branch = decision.candidate.metadata.get("branch", BRANCH_C)
            groups.setdefault(branch, []).append(decision)
        for entries in groups.values():
            entries.sort(key=lambda item: item.weighted_score, reverse=True)
        return groups

    def _weight_decisions(self, decisions: Sequence[SnippetDecision]) -> List[tuple[float, SnippetDecision]]:
        weighted: List[tuple[float, SnippetDecision]] = []
        for decision in decisions:
            branch = decision.candidate.metadata.get("branch", BRANCH_C)
            branch_conf = self.branch_metrics.get(branch, {}).get("confidence", DEFAULT_CONFIDENCE)
            weight = max(0.05, decision.weighted_score) * branch_conf
            weighted.append((weight, decision))
        weighted.sort(key=lambda item: item[0], reverse=True)
        return weighted


def branch_candidate_metadata(chunk: common.Chunk, snippet: common.Snippet, branch_label: str) -> Dict[str, object]:
    metadata = {
        "branch": branch_label,
        "page": chunk.page,
        "region": (chunk.metadata or {}).get("region_type"),
    }
    notes = snippet.notes or {}
    strategy = notes.get("branch_strategy")
    if strategy:
        metadata["strategy"] = strategy
    return metadata


__all__ = [
    "load_branch_manifest",
    "build_branch_context",
    "extra_branch_candidates",
    "annotate_candidate",
    "MultiBranchFusionEngine",
    "branch_candidate_metadata",
]
