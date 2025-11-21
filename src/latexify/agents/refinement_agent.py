"""Iterative refinement agent that uses compilation feedback to fix LaTeX."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

from ..pipeline.compilation_feedback import CompilationFeedbackAnalyzer
from .graph_state import GraphState

LOGGER = logging.getLogger(__name__)


@dataclass
class CompilationPass:
    attempt: int
    success: bool
    issues: List[str]
    actions: List[Dict[str, str]]


class RefinementAgent:
    """Run LaTeX compilation multiple times with feedback-driven fixes."""

    def __init__(self, max_passes: int = 3, *, compile_callable: Callable[[Path], Path | None] | None = None) -> None:
        self.max_passes = max_passes
        self.analyzer = CompilationFeedbackAnalyzer()
        self.state = GraphState(chunk_id="latex-refinement", content="")
        self._injected_packages: set[str] = set()
        self._defined_commands: set[str] = set()
        self._compile_callable = compile_callable

    def refine(self, tex_path: Path, *, max_passes: int | None = None) -> Dict[str, object]:
        passes: List[CompilationPass] = []
        requested_passes = max_passes or self.max_passes
        for attempt in range(1, requested_passes + 1):
            pdf_path = self._compile(tex_path)
            success = pdf_path is not None
            log_path = tex_path.with_suffix(".log")
            report = self.analyzer.analyze(log_path)
            actions = [] if success else self.analyzer.plan_actions(report.issues)
            passes.append(
                CompilationPass(
                    attempt=attempt,
                    success=success,
                    issues=[issue.code for issue in report.issues],
                    actions=actions,
                )
            )
            stage_note = "success" if success else "retry"
            self.state.mark_stage("refinement", notes=f"{stage_note}:{attempt}")
            self.state.record_metrics(refinement_attempts=attempt)
            self.state.log(f"attempt {attempt}: success={success} issues={passes[-1].issues}")
            if success:
                break
            if not actions:
                break
            self._apply_actions(tex_path, actions)
        summary = {
            "success": passes[-1].success if passes else False,
            "passes": [
                {
                    "attempt": entry.attempt,
                    "success": entry.success,
                    "issues": entry.issues,
                    "actions": entry.actions,
                }
                for entry in passes
            ],
        }
        summary["history"] = list(self.state.history)
        summary["stage_history"] = list(self.state.stage_history)
        summary["metrics"] = dict(self.state.metrics)
        return summary

    def _compile(self, tex_path: Path) -> Path | None:
        if self._compile_callable is None:
            raise RuntimeError("RefinementAgent compile callable is not configured")
        return self._compile_callable(tex_path)

    def _apply_actions(self, tex_path: Path, actions: List[Dict[str, str]]) -> None:
        for action in actions:
            action_type = action.get("type")
            if action_type == "add-package":
                package = action.get("package")
                if package:
                    self._inject_package(tex_path, package)
            elif action_type == "define-command":
                command = action.get("command")
                if command:
                    self._define_command(tex_path, command)
            elif action_type == "math-mode-fix":
                self._normalize_math(tex_path)

    def _inject_package(self, tex_path: Path, package: str) -> None:
        if package in self._injected_packages:
            return
        text = tex_path.read_text(encoding="utf-8")
        if f"\\usepackage{{{package}}}" in text or f"\\usepackage[{package}]" in text:
            return
        insert_idx = text.find("\\begin{document}")
        if insert_idx == -1:
            return
        before = text[:insert_idx]
        after = text[insert_idx:]
        before += f"\\usepackage{{{package}}}\n"
        tex_path.write_text(before + after, encoding="utf-8")
        self._injected_packages.add(package)
        LOGGER.info("Inserted missing package %s", package)

    def _define_command(self, tex_path: Path, command: str) -> None:
        if command in self._defined_commands:
            return
        text = tex_path.read_text(encoding="utf-8")
        macro = f"\\providecommand{{\\{command}}}{{\\texttt{{\\{command}}}}}"
        if macro in text:
            return
        insert_idx = text.find("\\begin{document}")
        if insert_idx == -1:
            return
        before = text[:insert_idx]
        after = text[insert_idx:]
        before += macro + "\n"
        tex_path.write_text(before + after, encoding="utf-8")
        self._defined_commands.add(command)
        LOGGER.info("Defined fallback for \\%s", command)

    def _normalize_math(self, tex_path: Path) -> None:
        text = tex_path.read_text(encoding="utf-8")
        if "$$" not in text and "\\begin{align}" not in text:
            return
        replaced = self._convert_double_dollar(text)
        replaced = replaced.replace("\\begin{align}", "\\begin{aligned}")
        replaced = replaced.replace("\\end{align}", "\\end{aligned}")
        tex_path.write_text(replaced, encoding="utf-8")
        LOGGER.info("Normalized math delimiters for %s", tex_path.name)

    def _convert_double_dollar(self, text: str) -> str:
        parts = text.split("$$")
        if len(parts) <= 1:
            return text
        rebuilt: List[str] = []
        open_delim = True
        for idx, part in enumerate(parts):
            rebuilt.append(part)
            if idx == len(parts) - 1:
                break
            rebuilt.append("\\[" if open_delim else "\\]")
            open_delim = not open_delim
        if not open_delim:
            rebuilt.append("\\]")
        return "".join(rebuilt)


__all__ = ["RefinementAgent", "CompilationPass"]
