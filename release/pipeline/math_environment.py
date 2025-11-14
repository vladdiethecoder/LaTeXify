"""Detect and normalize LaTeX math environments."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict


INLINE_MATH_RE = re.compile(r"\$(?!\$).+?\$")
DISPLAY_ENV_RE = re.compile(r"\\begin\{(?P<env>[a-z*]+)\}")
ALIGN_HINT_RE = re.compile(r"(=|\\approx|\\sim).*(\\\\|&)")
CASES_HINT_RE = re.compile(r"(cases|piecewise)", re.IGNORECASE)
MATRIX_HINT_RE = re.compile(r"\\begin\{[pbv]?matrix\}")


@dataclass
class MathEnvironmentDetector:
    """Heuristically pick the most appropriate LaTeX math environment."""

    prefer_align: bool = True

    def detect(self, block_type: str, text: str, metadata: Dict[str, object] | None = None) -> str | None:
        stripped = text.strip()
        if not stripped:
            return None
        existing = DISPLAY_ENV_RE.search(stripped)
        if existing:
            return existing.group("env")
        region = (metadata or {}).get("region_type")
        if region == "table":
            return None
        if MATRIX_HINT_RE.search(stripped):
            return "bmatrix"
        if CASES_HINT_RE.search(stripped):
            return "cases"
        if "&" in stripped or ALIGN_HINT_RE.search(stripped):
            return "align*" if self.prefer_align else "align"
        if "\\\\" in stripped:
            return "align*"
        if block_type in {"equation", "formula"} or INLINE_MATH_RE.search(stripped):
            return "equation"
        if stripped.startswith("\\left[") and "\\right]" in stripped:
            return "bmatrix"
        return None

    def wrap(self, block_type: str, text: str, metadata: Dict[str, object] | None = None) -> str:
        env = self.detect(block_type, text, metadata)
        if not env:
            return text
        stripped = text.strip()
        if stripped.startswith("\\begin"):
            return text
        body = stripped
        if env.endswith("*") and not env.startswith("align"):
            env = env.rstrip("*")
        return f"\\begin{{{env}}}\n{body}\n\\end{{{env}}}"


__all__ = ["MathEnvironmentDetector"]
