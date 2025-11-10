from __future__ import annotations

"""Lightweight critic agent placeholder for deterministic orchestration tests."""

import re
import shutil
import subprocess
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from .specialist_router import SpecialistDecision

LATEXMK_BIN = shutil.which("latexmk")
LATEXMK_ARGS = [
    "-halt-on-error",
    "-interaction=nonstopmode",
    "-pdf",
    "-quiet",
]


def _extract_max_attempts(source: Dict | None) -> Optional[int]:
    if not isinstance(source, dict):
        return None
    candidates: List[Optional[int]] = []
    for key in ("critic", "review", "synthesis"):
        sub = source.get(key)
        if isinstance(sub, dict):
            for field in ("max_attempts", "max_retries"):
                val = sub.get(field)
                if isinstance(val, int) and val > 0:
                    candidates.append(val)
    for field in (
        "critic_max_attempts",
        "review_max_attempts",
        "max_attempts",
        "max_retries",
    ):
        val = source.get(field)
        if isinstance(val, int) and val > 0:
            candidates.append(val)
    for value in candidates:
        if value is not None:
            return value
    return None


@dataclass(frozen=True)
class ReviewResult:
    accepted: bool
    feedback: str = ""


class CriticAgent:
    """Critic that verifies snippets via latexmk or heuristic fallbacks."""

    def __init__(
        self,
        plan: Dict | None = None,
        *,
        compiler: Optional[Callable[[str], Tuple[bool, str]]] = None,
    ):
        default_attempts = _extract_max_attempts(plan) or 1
        self._default_attempts = max(1, int(default_attempts))
        self._compiler = compiler or self._run_latexmk

    def max_attempts(self, task: Dict | None = None) -> int:
        attempt_override = _extract_max_attempts(task)
        if isinstance(attempt_override, int) and attempt_override > 0:
            return attempt_override
        return self._default_attempts

    def review(
        self,
        snippet: str,
        *,
        bundle: Dict,
        decision: SpecialistDecision,
        attempt: int,
        feedback_history: Iterable[str],
    ) -> ReviewResult:
        """Verify snippet and emit actionable feedback when it fails."""

        compile_ok, compile_log = self._compiler(snippet)
        flag_messages = self._detect_placeholder_flags(snippet)
        issues: List[str] = []
        if not compile_ok:
            issues.append(f"Compilation failed ({compile_log.strip() or 'unknown error'}).")
        issues.extend(flag_messages)
        if not issues:
            return ReviewResult(accepted=True, feedback="")

        remaining = max(self.max_attempts(bundle) - attempt, 0)
        hint = f"{remaining} attempt(s) left." if remaining > 0 else "No attempts remain; escalate."
        feedback = " ".join(issues + [hint])
        return ReviewResult(accepted=False, feedback=feedback)

    def _detect_placeholder_flags(self, snippet: str) -> List[str]:
        issues: List[str] = []
        if re.search(r"TODO|\\todo|\\placeholder", snippet, re.IGNORECASE):
            issues.append("Remove TODO/placeholder markers before submission.")
        if re.search(r"\?\?\?", snippet):
            issues.append("Replace '???' markers with real values.")
        return issues

    def _run_latexmk(self, snippet: str) -> Tuple[bool, str]:
        if not LATEXMK_BIN:
            return True, "latexmk_not_available"
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            tex_path = tmp_path / "snippet.tex"
            wrapped = textwrap.dedent(
                r"""
                \documentclass{article}
                \usepackage{amsmath}
                \usepackage{graphicx}
                \usepackage{booktabs}
                \begin{document}
                """
            ).strip() + "\n" + snippet + "\n\\end{document}\n"
            tex_path.write_text(wrapped, encoding="utf-8")
            try:
                subprocess.run(
                    [LATEXMK_BIN, *LATEXMK_ARGS, tex_path.name],
                    cwd=tmp_path,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=30,
                )
                return True, ""
            except subprocess.CalledProcessError as exc:
                log_excerpt = ""
                log_file = tmp_path / "snippet.log"
                if log_file.exists():
                    log_excerpt = log_file.read_text(encoding="utf-8")[-400:].strip()
                err = log_excerpt or exc.stderr.decode("utf-8", errors="ignore")
                return False, err
            except Exception as exc:  # pragma: no cover - defensive
                return False, str(exc)
