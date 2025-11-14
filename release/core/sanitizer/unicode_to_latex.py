"""Unicode → LaTeX normalization utilities."""
from __future__ import annotations

from typing import Dict

# Known multi-character sequences that should map to specific LaTeX commands.
SEQUENCE_LATEX_MAP: Dict[str, str] = {
    "∉": r"\notin",
    "⊈": r"\nsubseteq",
    "⊄": r"\nsubset",
    "≱": r"\ngeq",
    "≰": r"\nleq",
}

# Expandable lookup for troublesome Unicode glyphs that frequently leak through.
# Entries should remain context-free so they are safe in both math and text mode.
UNICODE_LATEX_MAP: Dict[str, str] = {
    "−": "-",  # minus sign
    "–": "--",
    "—": "---",
    "∈": r"\in",
    "∉": r"\notin",
    "∪": r"\cup",
    "∩": r"\cap",
    "∞": r"\infty",
    "⇒": r"\Rightarrow",
    "→": r"\rightarrow",
    "←": r"\leftarrow",
    "≤": r"\leq",
    "≥": r"\geq",
    "≠": r"\neq",
    "≈": r"\approx",
    "±": r"\pm",
    "∑": r"\sum",
    "∏": r"\prod",
    "×": r"\times",
    "√": r"\sqrt{}",
    "′": r"^{\prime}",
    "″": r"^{\prime\prime}",
    "̸": r"\not",  # combining long solidus overlay (fallback)
}


def sanitize_unicode_to_latex(payload: str) -> str:
    """Replace problematic Unicode glyphs with portable LaTeX macros."""

    if not payload:
        return payload
    sanitized = payload
    for needle, replacement in SEQUENCE_LATEX_MAP.items():
        sanitized = sanitized.replace(needle, replacement)
    for needle, replacement in UNICODE_LATEX_MAP.items():
        sanitized = sanitized.replace(needle, replacement)
    return sanitized
