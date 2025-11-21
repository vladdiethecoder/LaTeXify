"""LLM-backed normalization for OCR'd equations."""
from __future__ import annotations

import logging

from ..core.sanitizer.unicode_to_latex import sanitize_unicode_to_latex
from ..models.vllm_client import get_vllm_client

LOGGER = logging.getLogger(__name__)
PROMPT = """You normalize OCR equations. Convert the input into canonical LaTeX.
Replace any unicode math or currency characters (e.g., ¢) with proper LaTeX commands.
Respond ONLY with LaTeX.

Equation:
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
            response = self._client.generate(prompt, stop=[], max_tokens=256)
        except Exception as exc:  # pragma: no cover - runtime guard
            LOGGER.debug("Equation normalizer LLM failed: %s", exc)
            return sanitized
        cleaned = response.strip()
        if cleaned.lower().startswith("<latex>"):
            cleaned = cleaned.split("</latex>", 1)[0]
            cleaned = cleaned.replace("<latex>", "")
        return cleaned.strip() or sanitized


equation_normalizer = EquationNormalizer()


__all__ = ["equation_normalizer"]
