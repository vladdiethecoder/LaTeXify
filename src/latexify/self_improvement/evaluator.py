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
        
        solved = 1 if result.passed else 0
        attempted = 1
        
        # Continuous scoring for better gradient
        score = float(solved)
        if result.passed:
            # Analyze logs for quality
            try:
                # Locate log file from smoke run (hardcoded assumption based on test_smoke_release.py)
                log_path = self.validator.workdir / "src/latexify/samples/build/main.log" # Adjust path if needed
                # The smoke test in test_smoke_release uses tmp_path, so finding the log is tricky without parsing stdout.
                # However, run_release might leave artifacts or we can parse the pytest stdout if it printed paths.
                
                # Actually, simpler: parse the pytest stdout for "LaTeX Warning" or "error".
                errors = result.output.lower().count("error")
                warnings = result.output.lower().count("warning")
                
                # Penalize
                penalty = (errors * 0.05) + (warnings * 0.01)
                score = max(0.1, 1.0 - penalty)
            except Exception:
                pass

        metrics = AgentMetrics(
            score=score,
            tasks_attempted=attempted,
            tasks_solved=solved,
            failed_tasks={tests[0]: result.output} if not result.passed else {},
            passed_tasks=tests if result.passed else [],
        )
        agent.metrics = metrics
        return agent
