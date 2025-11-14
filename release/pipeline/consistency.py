"""Symbolic math consistency checks for generated LaTeX."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Set

try:  # pragma: no cover - optional heavy dependency
    import sympy as sp  # type: ignore
except Exception:  # pragma: no cover
    sp = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from latex2sympy2 import latex2sympy  # type: ignore
except Exception:  # pragma: no cover
    latex2sympy = None  # type: ignore

VARIABLE_RE = re.compile(r"[A-Za-z]\\w*")
OPERATOR_SET = {"+", "-", "*", "/", "^", "=", r"\times", r"\cdot"}


@dataclass
class MathConsistencyValidator:
    """Compare symbolic content between source OCR text and generated LaTeX."""

    def variable_overlap(self, source: str, generated: str) -> float:
        source_vars = self._extract_variables(source)
        generated_vars = self._extract_variables(generated)
        if not source_vars or not generated_vars:
            return 0.0
        intersection = len(source_vars & generated_vars)
        union = len(source_vars | generated_vars)
        return intersection / union if union else 0.0

    def operator_overlap(self, source: str, generated: str) -> float:
        source_ops = self._extract_operators(source)
        generated_ops = self._extract_operators(generated)
        if not source_ops or not generated_ops:
            return 0.0
        intersection = len(source_ops & generated_ops)
        union = len(source_ops | generated_ops)
        return intersection / union if union else 0.0

    def structure_similarity(self, source: str, generated: str) -> float:
        lhs = self._to_sympy(source)
        rhs = self._to_sympy(generated)
        if lhs is None or rhs is None:
            return 0.0
        try:  # pragma: no cover - sympy heavy
            simplified = sp.simplify(lhs - rhs) if sp is not None else None
            if simplified == 0:
                return 1.0
        except Exception:
            return 0.0
        return 0.0

    def validate(self, source: str, generated: str) -> Dict[str, float]:
        return {
            "symbol_overlap": round(self.variable_overlap(source, generated), 3),
            "operator_overlap": round(self.operator_overlap(source, generated), 3),
            "structure_similarity": round(self.structure_similarity(source, generated), 3),
        }

    def _extract_variables(self, text: str) -> Set[str]:
        return {match.group(0) for match in VARIABLE_RE.finditer(text or "")}

    def _extract_operators(self, text: str) -> Set[str]:
        operators = set()
        for op in OPERATOR_SET:
            if op in text:
                operators.add(op)
        return operators

    def _to_sympy(self, text: str):
        if not text.strip():
            return None
        if latex2sympy is not None:
            try:
                return latex2sympy(text)
            except Exception:
                pass
        if sp is not None:
            try:
                return sp.sympify(text)
            except Exception:
                return None
        return None


__all__ = ["MathConsistencyValidator"]
