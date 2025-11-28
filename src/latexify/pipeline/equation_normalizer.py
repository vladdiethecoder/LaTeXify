"""LLM-backed normalization for OCR'd equations."""
from __future__ import annotations

import logging
from typing import List

from ..core.sanitizer.unicode_to_latex import sanitize_unicode_to_latex
from ..models.vllm_client import get_vllm_client

LOGGER = logging.getLogger(__name__)
PROMPT = """You are an expert LaTeX typesetter. Normalize the following OCR math snippet into canonical LaTeX.

Rules:
1. Replace unicode math/currency chars (e.g., ¢, ×, ÷) with proper macros.
2. If the input contains multiple lines of equations that should be aligned (e.g. a system of equations, or a derivation steps), wrap them in an `align*` environment and use `&` for alignment (usually at the equals sign).
3. If it is a single display equation, wrap it in `equation*` or `\[ ... \]`.
4. If it is inline math text, just return the math content without wrappers, or wrap in `$ ... $` if context implies inline.
5. Do NOT add preamble or explanations. Respond ONLY with the LaTeX code.

Input:
{equation}
"""


class EquationNormalizer:
    """Uses a lightweight LLM to standardize math snippets."""

    def __init__(self) -> None:
        self._client = get_vllm_client()

    def normalize(self, text: str) -> str:
        sanitized = sanitize_unicode_to_latex(text)
        if not sanitized.strip():
            return sanitized
        if self._client is None:
            return sanitized
        prompt = PROMPT.format(equation=sanitized.strip())
        try:
            response = self._client.generate(prompt, stop=[], max_tokens=512)
        except Exception as exc:  # pragma: no cover - runtime guard
            LOGGER.debug("Equation normalizer LLM failed: %s", exc)
            return sanitized
        cleaned = response.strip()
        # Remove markdown code blocks if present
        if cleaned.startswith("```latex"):
            cleaned = cleaned.replace("```latex", "").replace("```", "")
        elif cleaned.startswith("```"):
            cleaned = cleaned.replace("```", "")
            
        if cleaned.lower().startswith("<latex>"):
            cleaned = cleaned.split("</latex>", 1)[0]
            cleaned = cleaned.replace("<latex>", "")
        normalized = cleaned.strip() or sanitized
        return normalized


equation_normalizer = EquationNormalizer()


__all__ = ["equation_normalizer"]
