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
    "−": "-",  # minus sign (U+2212) -> hyphen (U+002D)
    "\u2212": "-", # Explicit unicode escape for minus sign
    "–": "--",
    "—": "---",
    "“": "``",
    "”": "''",
    "‘": "`",
    "’": "'",
    "‚": ",",
    "„": ",,",
    "™": "(TM)",
    "¢": r"\textcent",
    "£": r"\pounds",
    "¥": r"\textyen",
    "�": "?",  # replacement character
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
    "²": r"^{2}",
    "¹": r"^{1}",
    "⁻": r"^{-}",
    "·": r"\cdot",
    "γ": r"\gamma",
    "€": r"\texteuro",
    "«": "<<",
    "»": ">>",
    "‹": "<",
    "›": ">",
    "…": r"\ldots",
    "•": r"\textbullet",
    "°": r"^{\circ}",
    "ℝ": r"\mathbb{R}",
    "©": "(C)",
}

CONTROL_CHAR_TRANSLATION = {code: " " for code in range(0, 32)}
for code in (9, 10, 13):
    CONTROL_CHAR_TRANSLATION.pop(code, None)
CONTROL_CHAR_TRANSLATION[0x7F] = " "


def sanitize_unicode_to_latex(payload: str) -> str:
    """Replace problematic Unicode glyphs with portable LaTeX macros."""

    if not payload:
        return payload
    sanitized = payload.translate(CONTROL_CHAR_TRANSLATION)
    for needle, replacement in SEQUENCE_LATEX_MAP.items():
        sanitized = sanitized.replace(needle, replacement)
    for needle, replacement in UNICODE_LATEX_MAP.items():
        sanitized = sanitized.replace(needle, replacement)
    return sanitized
