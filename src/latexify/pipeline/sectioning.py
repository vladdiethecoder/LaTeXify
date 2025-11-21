"""LLM-backed semantic sectioning utilities."""
from __future__ import annotations

import json
import logging
from textwrap import shorten
from typing import List, Sequence

from ..models.vllm_client import get_vllm_client

LOGGER = logging.getLogger(__name__)
PROMPT_TEMPLATE = """You segment paragraphs into coherent sections.
Each paragraph is prefixed with [index]. Group consecutive paragraphs into sections so that
each section is self-contained and under {max_chars} characters of total text.
Respond ONLY with JSON like [[1,2,3],[4,5],...].

Paragraphs:
{paragraphs}
"""
MAX_PARAGRAPHS = 80
TRUNCATE_CHARS = 220


class LLMSectioner:
    """Uses a lightweight LLM to propose logical section boundaries."""

    def __init__(self) -> None:
        self._client = get_vllm_client()
        if self._client is None:
            raise RuntimeError("vLLM client unavailable for sectioning.")

    def _format_paragraphs(self, paragraphs: Sequence[str]) -> str:
        lines = []
        for idx, para in enumerate(paragraphs[:MAX_PARAGRAPHS], start=1):
            snippet = shorten(para.replace("\n", " "), width=TRUNCATE_CHARS, placeholder=" ...")
            lines.append(f"[{idx}] {snippet}")
        return "\n".join(lines)

    def plan(self, paragraphs: Sequence[str], max_chars: int) -> List[List[int]]:
        if not paragraphs:
            return []
        prompt = PROMPT_TEMPLATE.format(
            max_chars=max(512, max_chars),
            paragraphs=self._format_paragraphs(paragraphs),
        )
        try:
            raw = self._client.generate(prompt, stop=[], max_tokens=384).strip()
        except Exception as exc:
            LOGGER.debug("LLM sectioner failed: %s", exc)
            return []
        if raw.lower().startswith("<latex>"):
            raw = raw.split("</latex>", 1)[0]
        try:
            data = json.loads(raw)
        except Exception:
            LOGGER.debug("LLM sectioner returned non-JSON output: %s", raw[:120])
            return []
        groups: List[List[int]] = []
        for entry in data:
            if isinstance(entry, list):
                indices = []
                for value in entry:
                    if isinstance(value, int) and value >= 1:
                        indices.append(value)
                if indices:
                    groups.append(indices)
        return groups


def build_sectioner() -> LLMSectioner | None:
    try:
        return LLMSectioner()
    except Exception as exc:
        LOGGER.info("Semantic LLM sectioner unavailable: %s", exc)
        return None


__all__ = ["LLMSectioner", "build_sectioner"]
