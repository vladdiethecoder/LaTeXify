"""LLM-assisted code block detection for LaTeX snippets."""
from __future__ import annotations

import logging
import re

from ..models.vllm_client import get_vllm_client

LOGGER = logging.getLogger(__name__)
PROMPT = """You convert markdown/plaintext into LaTeX verbatim blocks when it is clearly source code.
Wrap each code-only block in \\begin{verbatim} ... \\end{verbatim}. Do not alter existing LaTeX commands.
If there is no code, return the text unchanged. Respond with LaTeX only.

Snippet:
{snippet}
"""
FALLBACK_FENCE_RE = re.compile(r"```(.*?)```", re.DOTALL)


def wrap_code_blocks(text: str) -> str:
    snippet = text.strip()
    if not snippet:
        return text
    client = get_vllm_client()
    if client is None:
        return _fallback(text)
    prompt = PROMPT.format(snippet=snippet)
    try:  # pragma: no cover - depends on vLLM runtime
        response = client.generate(prompt, stop=[], max_tokens=512)
    except Exception as exc:
        LOGGER.debug("Code block detector LLM failed: %s", exc)
        return _fallback(text)
    cleaned = response.strip()
    if cleaned.lower().startswith("<latex>"):
        cleaned = cleaned.split("</latex>", 1)[0]
        cleaned = cleaned.replace("<latex>", "")
    return cleaned.strip() or text


def _fallback(text: str) -> str:
    def _replace(match: re.Match[str]) -> str:
        body = match.group(1).strip("\n")
        return f"\\begin{{verbatim}}\n{body}\n\\end{{verbatim}}"

    return FALLBACK_FENCE_RE.sub(_replace, text)


__all__ = ["wrap_code_blocks"]
