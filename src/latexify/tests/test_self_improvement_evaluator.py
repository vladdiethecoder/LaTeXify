from pathlib import Path

import sys
import types

sys.path.append(str(Path(__file__).resolve().parents[2]))

from latexify.self_improvement.evaluator import EvaluatorRunner, EvaluationConfig
from latexify.self_improvement.models import AgentVersion, AgentMetrics
from latexify.self_improvement.validator import ValidationResult, PatchValidator


class FakeValidator(PatchValidator):
    def __init__(self, passed: bool):
        self._passed = passed

    def run_pytest(self, tests, timeout=600):
        return ValidationResult(passed=self._passed, output="ok" if self._passed else "fail", returncode=0 if self._passed else 1)


def test_evaluator_records_success(tmp_path: Path):
    validator = FakeValidator(passed=True)
    evaluator = EvaluatorRunner(tmp_path, EvaluationConfig(tests=["dummy"]), validator=validator)
    agent = AgentVersion(version_id="v0", parent_id=None, strategy="VALIDATION", metrics=AgentMetrics())
    result = evaluator.evaluate(agent)
    assert result.metrics.score == 1.0
    assert result.metrics.tasks_solved == 1
    assert not result.metrics.failed_tasks


def test_evaluator_records_failure(tmp_path: Path):
    validator = FakeValidator(passed=False)
    evaluator = EvaluatorRunner(tmp_path, EvaluationConfig(tests=["dummy"]), validator=validator)
    agent = AgentVersion(version_id="v0", parent_id=None, strategy="VALIDATION", metrics=AgentMetrics())
    result = evaluator.evaluate(agent)
    assert result.metrics.score == 0.0
    assert result.metrics.failed_tasks
