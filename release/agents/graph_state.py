"""Shared dataclasses used by the experimental agent stack."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ContentChunk:
    """Lightweight representation of a chunk/page that needs layout reasoning."""

    chunk_id: str
    text: str
    metadata: dict[str, object] = field(default_factory=dict)



AGENT_PIPELINE_STAGE_MAP = {
    "creative": "branch_b_vision",
    "compile": "robust_compilation",
    "evaluator": "adaptive_quality_gate",
    "research": "branch_c_fusion",
    "refinement": "robust_compilation",
}


@dataclass
class GraphState:
    """Mutable state passed between agents, aligned with pipeline telemetry."""

    chunk_id: str
    content: str
    candidate_latex: Optional[str] = None
    failed_attempts: int = 0
    evaluation: Optional[str] = None
    score_notes: Optional[str] = None
    diagnostics: Optional[str] = None
    research_snippets: List[str] = field(default_factory=list)
    history: List[str] = field(default_factory=list)
    branch: Optional[str] = None
    stage_history: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    quality_report: Dict[str, object] = field(default_factory=dict)
    reward_report: Dict[str, object] = field(default_factory=dict)

    def log(self, message: str) -> None:
        self.history.append(message)

    def mark_stage(self, stage_key: str, notes: str | None = None) -> None:
        stage = AGENT_PIPELINE_STAGE_MAP.get(stage_key, stage_key)
        entry = stage if notes is None else f"{stage}:{notes}"
        self.stage_history.append(entry)

    def record_metrics(self, **values: float) -> None:
        self.metrics.update({k: float(v) for k, v in values.items()})

    def attach_artifact(self, name: str, path: str) -> None:
        self.artifacts[name] = path

    def set_quality_report(self, report: Dict[str, object]) -> None:
        self.quality_report = dict(report)

    def set_reward_report(self, report: Dict[str, object]) -> None:
        self.reward_report = dict(report)


__all__ = ["ContentChunk", "GraphState", "AGENT_PIPELINE_STAGE_MAP"]
