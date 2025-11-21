"""Pattern-based LaTeX compilation feedback analysis and fixes."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

LOGGER_NAME = "release.compilation_feedback"


@dataclass
class CompilationIssue:
    code: str
    message: str
    severity: str
    metadata: Dict[str, str] = field(default_factory=dict)
    actions: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class CompilationFeedbackReport:
    log_path: Path
    issues: List[CompilationIssue]

    def to_json(self) -> Dict[str, object]:
        return {
            "log_path": str(self.log_path),
            "issues": [
                {
                    "code": issue.code,
                    "message": issue.message,
                    "severity": issue.severity,
                    "metadata": issue.metadata,
                    "actions": issue.actions,
                }
                for issue in self.issues
            ],
        }


class CompilationFeedbackAnalyzer:
    """Recognize common LaTeX compilation errors and propose fixes."""

    MISSING_PACKAGE_RE = re.compile(r"File `([^`]+)\.sty' not found")
    UNDEFINED_CMD_RE = re.compile(r"Undefined control sequence\s*(?:\\(?P<cmd>[A-Za-z@]+))?")
    CMD_FOLLOWUP_RE = re.compile(r"\\[A-Za-z@]+")
    MATH_MODE_RE = re.compile(r"Missing \$ inserted|Bad math environment delimiter|Display math should end with|Illegal math")
    ALIGN_ERROR_RE = re.compile(r"Extra alignment tab has been changed to\\cr")
    PACKAGE_ENV_RE = re.compile(r"Environment ([A-Za-z*]+) undefined")

    def analyze(self, log_path: Path) -> CompilationFeedbackReport:
        if not log_path.exists():
            return CompilationFeedbackReport(log_path, [])
        text = log_path.read_text(errors="ignore")
        issues: List[CompilationIssue] = []
        issues.extend(self._detect_missing_packages(text))
        issues.extend(self._detect_undefined_commands(text))
        issues.extend(self._detect_environment_errors(text))
        issues.extend(self._detect_math_mode(text))
        issues.extend(self._detect_alignment(text))
        return CompilationFeedbackReport(log_path=log_path, issues=issues)

    def plan_actions(self, issues: Sequence[CompilationIssue]) -> List[Dict[str, str]]:
        actions: List[Dict[str, str]] = []
        seen = set()
        for issue in issues:
            for action in issue.actions:
                key = json.dumps(action, sort_keys=True)
                if key in seen:
                    continue
                seen.add(key)
                actions.append(action)
        return actions

    def _detect_missing_packages(self, text: str) -> List[CompilationIssue]:
        issues: List[CompilationIssue] = []
        for match in self.MISSING_PACKAGE_RE.finditer(text):
            package = match.group(1)
            message = f"missing package {package}"
            actions = [{"type": "add-package", "package": package}]
            issues.append(
                CompilationIssue(
                    code="missing-package",
                    message=message,
                    severity="fatal",
                    metadata={"package": package},
                    actions=actions,
                )
            )
        return issues

    def _detect_undefined_commands(self, text: str) -> List[CompilationIssue]:
        issues: List[CompilationIssue] = []
        for match in self.UNDEFINED_CMD_RE.finditer(text):
            cmd = match.group("cmd")
            if not cmd:
                snippet = text[match.end() : match.end() + 120]
                follow = self.CMD_FOLLOWUP_RE.search(snippet)
                cmd = follow.group(0)[1:] if follow else None
            if not cmd:
                continue
            message = f"undefined control sequence \\{cmd}"
            actions = [{"type": "define-command", "command": cmd}]
            issues.append(
                CompilationIssue(
                    code="undefined-command",
                    message=message,
                    severity="fatal",
                    metadata={"command": cmd},
                    actions=actions,
                )
            )
        return issues

    def _detect_environment_errors(self, text: str) -> List[CompilationIssue]:
        issues: List[CompilationIssue] = []
        for match in self.PACKAGE_ENV_RE.finditer(text):
            env = match.group(1)
            recommendation = self._package_for_environment(env)
            actions: List[Dict[str, str]] = []
            metadata = {"environment": env}
            if recommendation:
                metadata["package"] = recommendation
                actions.append({"type": "add-package", "package": recommendation})
            issues.append(
                CompilationIssue(
                    code="environment-undefined",
                    message=f"environment {env} undefined",
                    severity="fatal",
                    metadata=metadata,
                    actions=actions,
                )
            )
        return issues

    def _detect_math_mode(self, text: str) -> List[CompilationIssue]:
        if not self.MATH_MODE_RE.search(text):
            return []
        issue = CompilationIssue(
            code="math-mode",
            message="math mode delimiter mismatch",
            severity="fatal",
            metadata={},
            actions=[{"type": "math-mode-fix"}],
        )
        return [issue]

    def _detect_alignment(self, text: str) -> List[CompilationIssue]:
        if not self.ALIGN_ERROR_RE.search(text):
            return []
        issue = CompilationIssue(
            code="alignment-tabs",
            message="alignment tab mismatch",
            severity="warning",
            metadata={},
            actions=[{"type": "math-mode-fix"}],
        )
        return [issue]

    def _package_for_environment(self, env: str) -> str | None:
        mapping = {
            "align": "amsmath",
            "align*": "amsmath",
            "gather": "amsmath",
            "equation*": "amsmath",
            "tikzpicture": "tikz",
            "algorithm": "algorithm2e",
            "cases": "amsmath",
        }
        return mapping.get(env)


__all__ = ["CompilationFeedbackAnalyzer", "CompilationIssue", "CompilationFeedbackReport"]
