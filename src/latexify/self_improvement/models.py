from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Literal

StrategyChoice = Literal[
    "TARGETED",
    "EXPLORATORY",
    "SYNTHESIS",
    "VALIDATION",
    "CREATIVE",
    "CRITICAL",
]


@dataclass
class AgentMetrics:
    """Simple metrics holder for an agent version."""

    score: float = 0.0
    tasks_attempted: int = 0
    tasks_solved: int = 0
    failed_tasks: Dict[str, str] = field(default_factory=dict)  # task -> error
    passed_tasks: List[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if self.tasks_attempted == 0:
            return 0.0
        return self.tasks_solved / max(1, self.tasks_attempted)


@dataclass
class AgentVersion:
    """
    Represents a single agent snapshot that can enter the archive.
    """

    version_id: str
    parent_id: Optional[str]
    strategy: Optional[StrategyChoice]
    metrics: AgentMetrics = field(default_factory=AgentMetrics)
    summary: str = ""
    created_at: float = field(default_factory=lambda: time.time())
    code_path: Optional[Path] = None


@dataclass
class EvolutionConfig:
    """
    Runtime controls for the evolution loop.

    These can be adjusted dynamically by an LLM controller.
    """

    max_generations: int = 10
    exploration_temperature: float = 1.0
    retain_neutral: bool = True
    regression_tolerance: float = 0.02
    allow_parallel_candidates: int = 3
    thought_log_path: Path = Path("logs/self_improvement_thoughts.md")
    archive_limit: Optional[int] = None
