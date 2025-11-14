"""Unicode → LaTeX symbol normalization helpers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from ..core.sanitizer import sanitize_unicode_to_latex

DEFAULT_SYMBOL_MAP: Dict[str, str] = {
    "ℝ": r"\mathbb{R}",
    "ℂ": r"\mathbb{C}",
    "ℤ": r"\mathbb{Z}",
    "ℕ": r"\mathbb{N}",
    "→": r"\rightarrow",
    "↦": r"\mapsto",
    "⇔": r"\Leftrightarrow",
    "↔": r"\leftrightarrow",
    "⋮": r"\vdots",
    "⋯": r"\cdots",
    "·": r"\cdot",
    "∂": r"\partial",
    "∇": r"\nabla",
    "∠": r"\angle",
    "°": r"^{\circ}",
    "≈": r"\approx",
    "≅": r"\cong",
    "≡": r"\equiv",
    "∝": r"\propto",
    "↻": r"\circlearrowright",
}


@dataclass
class SymbolNormalizer:
    """Apply a configurable Unicode → LaTeX normalization map."""

    symbol_map: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_SYMBOL_MAP))

    def normalize(self, text: str) -> str:
        normalized = sanitize_unicode_to_latex(text)
        for token, replacement in self.symbol_map.items():
            normalized = normalized.replace(token, replacement)
        return normalized


__all__ = ["SymbolNormalizer"]
