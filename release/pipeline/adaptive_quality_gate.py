"""Adaptive quality gating that tunes thresholds by document type and difficulty."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from ..core import common

OVERRIDE_ENV = "LATEXIFY_QUALITY_OVERRIDE"


@dataclass
class DocumentProfile:
    doc_type: str
    math_ratio: float
    figure_ratio: float
    difficulty: float


@dataclass
class QualityGateResult:
    passed: bool
    payload: Dict[str, object]
    failed_dimensions: List[str]
    overrides_applied: bool
    recovery_actions: List[str]


BASE_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "math": {"semantic": 0.8, "content": 0.7, "structural": 0.75, "visual": 0.65, "reward": 0.1},
    "reference": {"semantic": 0.7, "content": 0.65, "structural": 0.7, "visual": 0.75, "reward": 0.05},
    "narrative": {"semantic": 0.65, "content": 0.7, "structural": 0.65, "visual": 0.6, "reward": 0.05},
}


def infer_document_profile(chunks_path: Path, plan_path: Path | None = None) -> DocumentProfile:
    chunks = common.load_chunks(chunks_path) if chunks_path.exists() else []
    total = max(1, len(chunks))
    math_chunks = sum(1 for chunk in chunks if (chunk.metadata or {}).get("region_type") in {"equation", "formula"})
    figure_chunks = sum(1 for chunk in chunks if (chunk.metadata or {}).get("region_type") == "figure")
    noise_scores = [float((chunk.metadata or {}).get("noise_score", 0.4)) for chunk in chunks if (chunk.metadata or {}).get("noise_score") is not None]
    difficulty = sum(noise_scores) / len(noise_scores) if noise_scores else 0.4
    math_ratio = math_chunks / total
    figure_ratio = figure_chunks / total
    if math_ratio >= 0.25:
        doc_type = "math"
    elif figure_ratio >= 0.3:
        doc_type = "reference"
    else:
        doc_type = "narrative"
    return DocumentProfile(doc_type=doc_type, math_ratio=round(math_ratio, 3), figure_ratio=round(figure_ratio, 3), difficulty=round(difficulty, 3))


class AdaptiveQualityGate:
    def __init__(self, profile: DocumentProfile, *, override: Optional[bool] = None) -> None:
        self.profile = profile
        if override is None:
            env = os.environ.get(OVERRIDE_ENV, "0").lower()
            self.override_requested = env in {"1", "true", "yes"}
        else:
            self.override_requested = override

    def evaluate(self, metrics: Dict[str, object]) -> QualityGateResult:
        thresholds = dict(BASE_THRESHOLDS.get(self.profile.doc_type, BASE_THRESHOLDS["narrative"]))
        relaxation = self._relaxation_factor(metrics)
        for key in thresholds:
            thresholds[key] = max(0.3, thresholds[key] - relaxation)
        scores = self._compute_scores(metrics)
        failed = [dimension for dimension, value in scores.items() if value < thresholds.get(dimension, 0.5)]
        overrides_applied = self.override_requested and bool(failed)
        passed = not failed or overrides_applied
        recovery_actions = self._plan_recovery(failed)
        payload = {
            "document_profile": self.profile.__dict__,
            "thresholds": thresholds,
            "scores": scores,
            "failed_dimensions": failed,
            "relaxation": relaxation,
            "overrides_applied": overrides_applied,
            "recovery_actions": recovery_actions,
            "passed": passed,
        }
        return QualityGateResult(passed=passed, payload=payload, failed_dimensions=failed, overrides_applied=overrides_applied, recovery_actions=recovery_actions)

    def _compute_scores(self, metrics: Dict[str, object]) -> Dict[str, float]:
        hallucination = metrics.get("hallucination") or {}
        flagged = (hallucination.get("flagged_count", 0) or 0) + (hallucination.get("claim_flag_count", 0) or 0)
        total = max(1, hallucination.get("total", 0) or 1)
        hallucination_score = max(0.0, 1.0 - flagged / total)
        validation = metrics.get("validation") or {}
        validation_score = 1.0 if validation.get("success") else 0.0
        reward_report = metrics.get("reward") or {}
        reward_score = float(reward_report.get("reward", 0.0))
        reward_score = _clamp(reward_score, -1.0, 1.0)
        cross_validation = metrics.get("cross_validation") or {}
        semantic_score = float((cross_validation.get("semantic") or {}).get("composite", 0.5))
        content_score = float((cross_validation.get("content") or {}).get("composite", 0.0))
        structural_score = float((cross_validation.get("structural") or {}).get("composite", 0.0))
        visual_score = float((cross_validation.get("visual") or {}).get("composite", 0.0))
        return {
            "semantic": round(semantic_score, 3),
            "content": round(content_score, 3),
            "structural": round(structural_score, 3),
            "visual": round(visual_score, 3),
            "reward": round(reward_score, 3),
            "hallucination": round(hallucination_score, 3),
            "validation": round(validation_score, 3),
        }

    def _relaxation_factor(self, metrics: Dict[str, object]) -> float:
        input_profile = metrics.get("input_profile") or {}
        tier = str(input_profile.get("tier", "normal")).lower()
        tier_bonus = 0.05 if tier in {"difficult", "degraded"} else 0.0
        difficulty = self.profile.difficulty
        reward = float((metrics.get("reward") or {}).get("reward", 0.0))
        if reward < 0:
            difficulty += 0.05
        return round(min(0.2, difficulty * 0.3 + tier_bonus), 3)

    def _plan_recovery(self, failed: List[str]) -> List[str]:
        actions: List[str] = []
        if not failed:
            return actions
        if "visual" in failed:
            actions.append("rerun_visual_regression")
        if "semantic" in failed or "content" in failed:
            actions.append("route_to_active_learning")
        if "structural" in failed:
            actions.append("enable_monkey_ocr")
        if "reward" in failed:
            actions.append("increase_refinement_passes")
        return actions


def _clamp(value: float, lower: float, upper: float) -> float:
    if upper == lower:
        return lower
    return max(lower, min(upper, value))


def save_gate_payload(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


__all__ = [
    "AdaptiveQualityGate",
    "DocumentProfile",
    "QualityGateResult",
    "infer_document_profile",
    "save_gate_payload",
]
