"""Semantic validation for LaTeX equations using SymPy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal

from sympy import SympifyError
from sympy.parsing.latex import parse_latex


COMMON_ERROR_PATTERNS = {
    "unbalanced-braces": "Detected mismatched curly braces.",
    "unbalanced-parentheses": "Detected mismatched parentheses or brackets.",
    "unmatched-left-right": "Found \\left without a matching \\right (or vice-versa).",
    "unknown-function": "Function name missing backslash (sin, cos, log, ...).",
    "parse-error": "SymPy failed to parse the expression.",
    "empty-equation": "Equation text is empty.",
}


@dataclass
class ValidationIssue:
    code: str
    message: str
    severity: Literal["warning", "error"] = "error"
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class ValidationReport:
    valid: bool
    confidence: float
    issues: List[ValidationIssue] = field(default_factory=list)
    normalized: str = ""


class EquationValidator:
    """Runs best-effort LaTeX validation before handing text to the synthesizer."""

    FUNCTION_TOKENS = ("sin", "cos", "tan", "cot", "sec", "csc", "log", "ln", "exp")
    OPERATORS = ("=", "\\leq", "\\geq", "\\approx", "\\sim")

    def __init__(self) -> None:
        pass

    def validate(self, latex: str) -> ValidationReport:
        stripped = latex.strip()
        issues: List[ValidationIssue] = []
        if not stripped:
            issues.append(self._issue("empty-equation"))
            return ValidationReport(valid=False, confidence=0.0, issues=issues, normalized=stripped)
        if not self._balanced_delimiters(stripped):
            issues.append(self._issue("unbalanced-braces"))
        if not self._balanced_parentheses(stripped):
            issues.append(self._issue("unbalanced-parentheses"))
        if not self._balanced_left_right(stripped):
            issues.append(self._issue("unmatched-left-right", severity="warning"))
        unknown_functions = self._unknown_functions(stripped)
        for func in unknown_functions:
            issues.append(
                ValidationIssue(
                    code="unknown-function",
                    message=f"Function '{func}' is missing a leading backslash.",
                    severity="warning",
                )
            )
        parse_ok = self._sympy_parse(stripped, issues)
        valid = parse_ok and not any(issue.severity == "error" for issue in issues)
        confidence = self._score(valid, issues)
        return ValidationReport(valid=valid, confidence=confidence, issues=issues, normalized=stripped)

    def _balanced_delimiters(self, text: str) -> bool:
        stack: List[str] = []
        mapping = {"{": "}", "[": "]"}
        for char in text:
            if char in mapping:
                stack.append(char)
            elif char in mapping.values():
                if not stack:
                    return False
                opener = stack.pop()
                if mapping[opener] != char:
                    return False
        return not stack

    def _balanced_parentheses(self, text: str) -> bool:
        count = 0
        for char in text:
            if char == "(":
                count += 1
            elif char == ")":
                count -= 1
            if count < 0:
                return False
        return count == 0

    def _balanced_left_right(self, text: str) -> bool:
        left = text.count("\\left")
        right = text.count("\\right")
        return left == right

    def _unknown_functions(self, text: str) -> List[str]:
        unknown: List[str] = []
        for token in self.FUNCTION_TOKENS:
            pattern = f"\\{token}"
            if token in text and pattern not in text:
                unknown.append(token)
        return unknown

    def _sympy_parse(self, text: str, issues: List[ValidationIssue]) -> bool:
        try:
            parse_latex(text)
            return True
        except (SympifyError, ValueError) as exc:
            issues.append(
                ValidationIssue(
                    code="parse-error",
                    message=f"SymPy parser error: {exc}",
                    severity="error",
                )
            )
            return False

    def _score(self, valid: bool, issues: List[ValidationIssue]) -> float:
        if not issues and valid:
            return 0.95
        error_count = sum(1 for issue in issues if issue.severity == "error")
        warning_count = sum(1 for issue in issues if issue.severity == "warning")
        penalty = error_count * 0.25 + warning_count * 0.1
        return max(0.0, 0.8 - penalty)

    def _issue(self, code: str, severity: Literal["warning", "error"] = "error") -> ValidationIssue:
        message = COMMON_ERROR_PATTERNS.get(code, code)
        return ValidationIssue(code=code, message=message, severity=severity)


__all__ = [
    "EquationValidator",
    "ValidationIssue",
    "ValidationReport",
    "COMMON_ERROR_PATTERNS",
]
