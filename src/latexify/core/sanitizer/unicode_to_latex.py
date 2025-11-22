"Unicode ">→</ LaTeX normalization utilities."
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
    # Basic Punctuation & Dashes
    "−": "-",  # minus sign (U+2212)
    "\u2212": "-",
    "–": "--", # en dash
    "—": "---", # em dash
    "“": "``",
    "”": "''",
    "‘": "`",
    "’": "'",
    "‚": ",",
    "„": ",,",
    "…": r"\ldots",
    "•": r"\textbullet",
    "·": r"\cdot",
    "°": r"^{\circ}",
    
    # Math Operators & Relations
    "×": r"\times",
    "÷": r"\div",
    "±": r"\pm",
    "∓": r"\mp",
    "≤": r"\leq",
    "≥": r"\geq",
    "≠": r"\neq",
    "≈": r"\approx",
    "≡": r"\equiv",
    "∝": r"\propto",
    "∞": r"\infty",
    "√": r"\sqrt{}",
    "∂": r"\partial",
    "∇": r"\nabla",
    "∆": r"\Delta",
    
    # Sets & Logic
    "∈": r"\in",
    "∉": r"\notin",
    "∋": r"\ni",
    "⊂": r"\subset",
    "⊃": r"\supset",
    "⊆": r"\subseteq",
    "⊇": r"\supseteq",
    "∪": r"\cup",
    "∩": r"\cap",
    "∅": r"\emptyset",
    "∀": r"\forall",
    "∃": r"\exists",
    "∄": r"\nexists",
    "∴": r"\therefore",
    "∵": r"\because",
    
    # Arrows
    "→": r"\rightarrow",
    "←": r"\leftarrow",
    "↔": r"\leftrightarrow",
    "⇒": r"\Rightarrow",
    "⇐": r"\Leftarrow",
    "⇔": r"\Leftrightarrow",
    "↑": r"\uparrow",
    "↓": r"\downarrow",
    
    # Greek (Common)
    "α": r"\alpha", "β": r"\beta", "γ": r"\gamma", "δ": r"\delta", "ε": r"\epsilon",
    "θ": r"\theta", "λ": r"\lambda", "μ": r"\mu", "π": r"\pi", "ρ": r"\rho",
    "σ": r"\sigma", "τ": r"\tau", "φ": r"\phi", "ω": r"\omega",
    "Δ": r"\Delta", "Γ": r"\Gamma", "Θ": r"\Theta", "Λ": r"\Lambda",
    "Π": r"\Pi", "Σ": r"\Sigma", "Φ": r"\Phi", "Ψ": r"\Psi", "Ω": r"\Omega",

    # Superscripts/Subscripts
    "²": r"^{2}",
    "³": r"^{3}",
    "¹": r"^{1}",
    "⁰": r"^{0}",
    "⁻": r"^{- }",
    "⁺": r"^{+ }",
    "½": r"\frac{1}{2}",
    
    # Currency & Misc
    "¢": r"\textcent",
    "£": r"\pounds",
    "¥": r"\textyen",
    "€": r"\texteuro",
    "©": r"\textcopyright",
    "®": r"\textregistered",
    "™": r"\texttrademark",
    "": "?", 
    "ℝ": r"\mathbb{R}",
    
    # Primes (Safe)
    "′": "'", 
    "″": "''",
    "‴": "'''",
    "«": "<<",
    "»": ">>",
    "‹": "<",
    "›": ">",
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