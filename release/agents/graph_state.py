"""Shared dataclasses used by the experimental agent stack."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ContentChunk:
    """Lightweight representation of a chunk/page that needs layout reasoning."""

    chunk_id: str
    text: str
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class GraphState:
    """Mutable state passed between agents."""

    chunk_id: str
    content: str
    candidate_latex: Optional[str] = None
    failed_attempts: int = 0
    evaluation: Optional[str] = None
    score_notes: Optional[str] = None
    diagnostics: Optional[str] = None
    research_snippets: List[str] = field(default_factory=list)
    history: List[str] = field(default_factory=list)

    def log(self, message: str) -> None:
        self.history.append(message)


__all__ = ["ContentChunk", "GraphState"]
