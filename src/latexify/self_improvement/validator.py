from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class ValidationResult:
    passed: bool
    output: str
    returncode: int


class PatchValidator:
    """
    Runs targeted pytest subsets to validate a child agent before archiving.
    """

    def __init__(self, workdir: Path):
        self.workdir = workdir

    def run_pytest(self, tests: Optional[List[str]] = None, timeout: int = 600) -> ValidationResult:
        tests = tests or []
        # Empty list means "skip heavy validation" for quick iterations.
        if not tests:
            return ValidationResult(passed=True, output="Validation skipped (no tests supplied)", returncode=0)

        cmd = ["python", "-m", "pytest", *tests]
        try:
            result = subprocess.run(
                cmd,
                cwd=self.workdir,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            output = result.stdout + "\n" + result.stderr
            return ValidationResult(passed=result.returncode == 0, output=output, returncode=result.returncode)
        except subprocess.TimeoutExpired as exc:
            return ValidationResult(passed=False, output=str(exc), returncode=124)
