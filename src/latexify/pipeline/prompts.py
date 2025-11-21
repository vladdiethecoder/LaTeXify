"""Domain-specific prompt templates used by the LLM refiner."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


DEFAULT_PROMPTS: Dict[str, str] = {
    "equation": (
        "You are DeepSeek-Coder-V2 tuned for LaTeX. Rewrite the math using \\begin{{{env}}} ... \\end{{{env}}}, "
        "preserve symbol ordering, convert informal fractions to \\frac, and align multi-line derivations "
        "with ampersands."
    ),
    "proof": (
        "Format the reasoning as a formal proof: wrap with \\begin{{proof}} ... \\end{{proof}}, introduce numbered steps "
        "when the source lists multiple claims, and keep every cited theorem inline using \\textit{{}}."
    ),
    "table": (
        "Reconstruct the structure as a booktabs table. Align math columns, promote the first line into a caption when "
        "available, and emit only valid LaTeX."
    ),
    "question": (
        "Produce a numbered question environment with labeled parts (a), (b), .... "
        "Preserve any inline math exactly and keep wording close to the source."
    ),
    "default": (
        "Generate TeX that mirrors the source semantics. Prefer \\paragraph blocks for prose, keep AMS math environments "
        "intact, and avoid inventing new structure beyond what the source provides."
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
