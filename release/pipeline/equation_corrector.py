"""Pattern-based auto-corrections for LaTeX equations."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .equation_validator import ValidationReport


@dataclass
class EquationCorrection:
    corrected: str
    confidence: float
    applied_rules: List[str] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)


class EquationCorrector:
    """Applies lightweight heuristics to repair malformed equations."""

    FUNCTION_TOKENS = ("sin", "cos", "tan", "log", "ln", "exp")
    OPERATOR_SPACING = re.compile(r"\s*([=+\-*/])\s*")

    def correct(self, latex: str, report: ValidationReport | None = None) -> Optional[EquationCorrection]:
        candidate = latex
        applied: List[str] = []

        balanced = self._close_brackets(candidate)
        if balanced != candidate:
            applied.append("balance-brackets")
            candidate = balanced

        frac_fixed = self._normalize_frac(candidate)
        if frac_fixed != candidate:
            applied.append("normalize-frac")
            candidate = frac_fixed

        func_normalized = self._normalize_functions(candidate)
        if func_normalized != candidate:
            applied.append("normalize-functions")
            candidate = func_normalized

        spaced = self._normalize_spacing(candidate)
        if spaced != candidate:
            applied.append("normalize-spacing")
            candidate = spaced

        if not applied or candidate.strip() == latex.strip():
            return None

        confidence = min(0.95, 0.5 + 0.1 * len(applied))
        metadata: Dict[str, object] = {}
        if report:
            metadata["issues"] = [issue.code for issue in report.issues]
            confidence = min(0.95, confidence + report.confidence * 0.1)
        return EquationCorrection(corrected=candidate, confidence=confidence, applied_rules=applied, metadata=metadata)

    def _close_brackets(self, latex: str) -> str:
        stack: List[str] = []
        mapping = {"{": "}", "(": ")", "[": "]"}
        tokens: List[str] = []
        idx = 0
        while idx < len(latex):
            char = latex[idx]
            if char in mapping:
                stack.append(char)
                tokens.append(char)
            elif char in mapping.values():
                if stack and mapping[stack[-1]] == char:
                    stack.pop()
                tokens.append(char)
            else:
                tokens.append(char)
            idx += 1
        while stack:
            opener = stack.pop()
            tokens.append(mapping[opener])
        left_count = latex.count("\\left")
        right_count = latex.count("\\right")
        if left_count > right_count:
            tokens.append("\\right.")
        elif right_count > left_count:
            tokens.insert(0, "\\left.")
        return "".join(tokens)

    def _normalize_frac(self, latex: str) -> str:
        def _replace(match: re.Match[str]) -> str:
            numerator = match.group(1).strip()
            denominator = match.group(2).strip()
            return f"\\frac{{{numerator}}}{{{denominator}}}"

        pattern = re.compile(r"\\frac\s+([^\s\\{}]+)\s+([^\s\\{}]+)")
        updated = pattern.sub(_replace, latex)
        inline_pattern = re.compile(r"\\frac\s*([0-9A-Za-z])([0-9A-Za-z])")

        def _split_digits(match: re.Match[str]) -> str:
            numerator = match.group(1)
            denominator = match.group(2)
            return f"\\frac{{{numerator}}}{{{denominator}}}"

        updated = inline_pattern.sub(_split_digits, updated)
        return updated

    def _normalize_functions(self, latex: str) -> str:
        def _replace(match: re.Match[str]) -> str:
            token = match.group(1)
            return f"\\{token}"

        pattern = re.compile(r"(?<!\\)\b(" + "|".join(self.FUNCTION_TOKENS) + r")\b")
        return pattern.sub(_replace, latex)

    def _normalize_spacing(self, latex: str) -> str:
        normalized = self.OPERATOR_SPACING.sub(lambda match: f" {match.group(1)} ", latex)
        normalized = re.sub(r"\s{2,}", " ", normalized)
        return normalized.strip()


__all__ = ["EquationCorrector", "EquationCorrection"]
