from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .models import AgentVersion, AgentMetrics
from .validator import PatchValidator


@dataclass
class EvaluationConfig:
    tests: List[str]
    timeout: int = 600


class EvaluatorRunner:
    """
    Runs a task suite (pytest-based) to produce AgentMetrics for a given version.
    """

    def __init__(self, repo_root: Path, config: EvaluationConfig, validator: Optional[PatchValidator] = None):
        self.config = config
        self.validator = validator or PatchValidator(repo_root)

    def evaluate(self, agent: AgentVersion) -> AgentVersion:
        tests = self.config.tests
        if not tests:
            agent.metrics = AgentMetrics(score=0.0, tasks_attempted=0, tasks_solved=0)
            return agent

        result = self.validator.run_pytest(tests, timeout=self.config.timeout)
        # Simple success metric: all tests must pass.
        solved = 1 if result.passed else 0
        attempted = 1
        metrics = AgentMetrics(
            score=solved / attempted,
            tasks_attempted=attempted,
            tasks_solved=solved,
            failed_tasks={tests[0]: result.output} if not result.passed else {},
            passed_tasks=tests if result.passed else [],
        )
        agent.metrics = metrics
        return agent
