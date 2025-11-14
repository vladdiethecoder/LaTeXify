"""Targeted LaTeX fixer that parses compilation logs."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

try:  # pragma: no cover - optional dependency
    from texoutparse import parser as tex_parser  # type: ignore
except Exception:  # pragma: no cover
    tex_parser = None  # type: ignore

LOGGER = logging.getLogger(__name__)
MISSING_DOLLAR_RE = re.compile(r"Missing \$ inserted")
UNDEFINED_CMD_RE = re.compile(r"Undefined control sequence\s*\\(?P<command>[A-Za-z]+)")
ALIGNMENT_RE = re.compile(r"Extra alignment tab has been changed to \\cr")
OVERFULL_RE = re.compile(r"Overfull \\hbox")


@dataclass
class LaTeXErrorRepair:
    """Parse latexmk/tectonic logs and apply targeted fixes."""

    max_passes: int = 2
    injected_preamble: str = "\\sloppy"
    patched_chunks: Dict[str, int] = field(default_factory=dict)

    def analyze_log(self, log_path: Path) -> List[str]:
        if not log_path.exists():
            return []
        if tex_parser is not None:  # pragma: no cover - optional dependency
            try:
                doc = tex_parser.parse(log_path.read_text(errors="ignore"))
                if doc and doc.errors:
                    return [err.message.lower() for err in doc.errors]
            except Exception:
                LOGGER.debug("texoutparse failed to inspect %s", log_path, exc_info=True)
        text = log_path.read_text(errors="ignore")
        issues: List[str] = []
        if MISSING_DOLLAR_RE.search(text):
            issues.append("missing-delimiter")
        if UNDEFINED_CMD_RE.search(text):
            issues.append("undefined-command")
        if ALIGNMENT_RE.search(text):
            issues.append("alignment")
        if OVERFULL_RE.search(text):
            issues.append("overfull")
        return issues

    def repair(self, tex_path: Path, log_path: Path) -> bool:
        issues = self.analyze_log(log_path)
        if not issues:
            return False
        content = tex_path.read_text(encoding="utf-8")
        updated = content
        if "missing-delimiter" in issues:
            updated = self._balance_delimiters(updated)
        if "undefined-command" in issues:
            updated = self._stub_unknown_commands(updated, log_path)
        if "alignment" in issues:
            updated = self._normalize_align(updated)
        if "overfull" in issues:
            updated = self._inject_sloppy(updated)
        if updated != content:
            tex_path.write_text(updated, encoding="utf-8")
            LOGGER.info("Applied LaTeXErrorRepair fixes (%s)", ", ".join(sorted(set(issues))))
            return True
        return False

    def _balance_delimiters(self, text: str) -> str:
        opened = text.count("{")
        closed = text.count("}")
        padding = "}" * max(0, opened - closed)
        return text + padding if padding else text

    def _stub_unknown_commands(self, text: str, log_path: Path) -> str:
        undefined: Sequence[str] = []
        match_iter = UNDEFINED_CMD_RE.finditer(log_path.read_text(errors="ignore"))
        undefined = [match.group("command") for match in match_iter if match.group("command")]
        patched = text
        for command in undefined:
            if f"\\newcommand{{\\{command}}}" in patched:
                continue
            patched = patched.replace(
                "\\begin{document}",
                f"\\newcommand{{\\{command}}}[1]{{#1}}\n\\begin{document}",
                1,
            )
        return patched

    def _normalize_align(self, text: str) -> str:
        return text.replace("\\begin{align}", "\\begin{aligned}").replace("\\end{align}", "\\end{aligned}")

    def _inject_sloppy(self, text: str) -> str:
        if self.injected_preamble in text:
            return text
        return text.replace("\\begin{document}", f"{self.injected_preamble}\n\\begin{document}", 1)


__all__ = ["LaTeXErrorRepair"]
