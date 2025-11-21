"""Markdown → LaTeX conversion backed by vLLM with regex fallback."""
from __future__ import annotations

import logging
import re
from typing import Callable

from ..models.vllm_client import get_vllm_client

LOGGER = logging.getLogger(__name__)
_FALLBACK_BOLD_RE = re.compile(r"(\*\*|__)(.+?)\1")
_FALLBACK_ITALIC_RE = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)")
_FALLBACK_UNDERSCORE_ITALIC_RE = re.compile(r"(?<!_)_(?!_)(.+?)(?<!_)_(?!_)")
_FALLBACK_CODE_RE = re.compile(r"`([^`]+)`")
_FALLBACK_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

LLM_PROMPT = """You convert Markdown into LaTeX while preserving structure. \
Respond only with LaTeX, wrapped between <latex> and </latex>.

Markdown:
{markdown}

<latex>
"""


def _fallback_convert(text: str) -> str:
    def _replace(pattern: re.Pattern[str], repl: Callable[[re.Match[str]], str], payload: str) -> str:
        return pattern.sub(repl, payload)

    converted = _replace(
        _FALLBACK_BOLD_RE,
        lambda m: r"\textbf{" + m.group(2) + "}",
        text,
    )
    converted = _replace(
        _FALLBACK_ITALIC_RE,
        lambda m: r"\textit{" + m.group(1) + "}",
        converted,
    )
    converted = _replace(
        _FALLBACK_UNDERSCORE_ITALIC_RE,
        lambda m: r"\textit{" + m.group(1) + "}",
        converted,
    )
    converted = _replace(
        _FALLBACK_CODE_RE,
        lambda m: r"\texttt{" + m.group(1) + "}",
        converted,
    )
    def _header_sub(match: re.Match[str]) -> str:
        level = len(match.group(1))
        label = match.group(2).strip()
        if level == 1:
            return r"\section{" + label + "}"
        if level == 2:
            return r"\subsection{" + label + "}"
        if level == 3:
            return r"\subsubsection{" + label + "}"
        return r"\paragraph{" + label + "}"

    converted = _FALLBACK_HEADER_RE.sub(_header_sub, converted)
    return converted


class MarkdownTranslator:
    """LLM-backed Markdown → LaTeX converter that falls back to regex heuristics."""

    def __init__(self) -> None:
        self._client = get_vllm_client()

    def convert(self, text: str) -> str:
        snippet = text.strip()
        if not snippet:
            return text
        if self._client is None:
            return _fallback_convert(text)
        prompt = LLM_PROMPT.format(markdown=snippet)
        try:
            response = self._client.generate(prompt, stop=["</latex>"], max_tokens=512)
        except Exception as exc:  # pragma: no cover - runtime guard
            LOGGER.debug("Markdown translator LLM failed: %s", exc)
            return _fallback_convert(text)
        cleaned = response.strip()
        if cleaned.lower().startswith("<latex>"):
            cleaned = cleaned[7:]
        return cleaned.strip() or _fallback_convert(text)


__all__ = ["MarkdownTranslator"]
