"""LLM-backed normalization for OCR'd equations."""
from __future__ import annotations

import logging
from typing import List

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
        normalized = cleaned.strip() or sanitized
        return self._align_multiline(normalized)

    def _align_multiline(self, payload: str) -> str:
        """Wrap multiline equations in an align* block with aligned equals signs."""
        lowered = payload.lower()
        if "\\begin{align" in lowered or "\\begin{aligned" in lowered or "\\begin{equation" in lowered:
            return payload
            
        if "\\question" in lowered or "question" in lowered or "\\section" in lowered:
            return payload
            
        lines: List[str] = [ln.strip() for ln in payload.splitlines() if ln.strip()]
        if len(lines) <= 1:
            return payload

        # Heuristic: Only align if it looks like a system of equations
        has_equals = any("=" in line for line in lines)
        has_math_chars = any(c in payload for c in ["\\", "^", "_", "+", "-"])
        
        if not (has_equals or has_math_chars):
            # Likely just text
            return payload

        aligned: List[str] = []
        for line in lines:
            if "=" in line and "&" not in line:
                head, tail = line.split("=", 1)
                aligned.append(f"{head.strip()} & = {tail.strip()}")
            else:
                aligned.append(line)

        return "\\begin{align*}\n" + " \\\\\n".join(aligned) + "\n\\end{align*}"


equation_normalizer = EquationNormalizer()


__all__ = ["equation_normalizer"]
