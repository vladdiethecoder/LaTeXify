"""Shared pydantic models used by the experimental agent stack."""
from __future__ import annotations

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, ConfigDict

AGENT_PIPELINE_STAGE_MAP = {
    "creative": "branch_b_vision",
    "compile": "robust_compilation",
    "evaluator": "adaptive_quality_gate",
    "research": "branch_c_fusion",
    "refinement": "robust_compilation",
}

class ContentChunk(BaseModel):
    """Lightweight representation of a chunk/page that needs layout reasoning."""
    model_config = ConfigDict(strict=True)

    chunk_id: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class GraphState(BaseModel):
    """Mutable state passed between agents, aligned with pipeline telemetry."""
    model_config = ConfigDict(strict=True, arbitrary_types_allowed=True)

    chunk_id: str
    content: str
    candidate_latex: Optional[str] = None
    failed_attempts: int = 0
    evaluation: Optional[str] = None
    score_notes: Optional[str] = None
    diagnostics: Optional[str] = None
    research_snippets: List[str] = Field(default_factory=list)
    history: List[str] = Field(default_factory=list)
    branch: Optional[str] = None
    stage_history: List[str] = Field(default_factory=list)
    metrics: Dict[str, float] = Field(default_factory=dict)
    artifacts: Dict[str, str] = Field(default_factory=dict)
    quality_report: Dict[str, Any] = Field(default_factory=dict)
    reward_report: Dict[str, Any] = Field(default_factory=dict)
    config: Dict[str, Any] = Field(default_factory=dict) # Added config for flexibility

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