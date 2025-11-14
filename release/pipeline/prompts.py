"""Domain-specific prompt templates used by the LLM refiner."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


DEFAULT_PROMPTS: Dict[str, str] = {
    "equation": (
        "Rewrite the math into canonical LaTeX with \\begin{{{env}}} ... \\end{{{env}}}, "
        "preserving the symbol order and ensuring each fraction becomes \\frac. "
        "Keep derivations aligned with ampersands."
    ),
    "proof": (
        "Format the reasoning as a proof using \\begin{{proof}} and numbered steps. "
        "Cite theorems inline using \\textit{{}} annotations when helpful."
    ),
    "table": (
        "Reconstruct the structure as a booktabs table with aligned math columns "
        "and concise captions."
    ),
    "question": (
        "Produce a numbered question environment with labeled parts (a), (b), ... "
        "and reserve inline math for short expressions."
    ),
    "default": (
        "Generate TeX that mirrors the source semantics using sentences, \\paragraph, "
        "and AMS math environments as needed."
    ),
}


@dataclass
class PromptLibrary:
    """Lookup library that returns tuned prompt strings for each content type."""

    prompts: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_PROMPTS))

    def get(self, content_type: str, **context: str) -> str:
        template = self.prompts.get(content_type, self.prompts["default"])
        return template.format(**context) if context else template


__all__ = ["PromptLibrary"]
