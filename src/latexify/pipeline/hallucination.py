"""LLM-based hallucination checks for section headers."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from textwrap import shorten
from typing import Dict, List

from ..core import common
from ..models.kimi_k2_adapter import get_kimi_adapter

LOGGER = logging.getLogger(__name__)
HALLUCINATION_TEMPERATURE = float(os.environ.get("LATEXIFY_HALLUCINATION_TEMPERATURE", "0.0"))
HALLUCINATION_MAX_TOKENS = int(os.environ.get("LATEXIFY_HALLUCINATION_MAX_TOKENS", "96"))
PROMPT = """Determine if the proposed section heading is supported by the source text.

Source:
{source}

Heading:
{heading}

Respond with YES if the heading clearly appears or is implied by the source, otherwise respond NO.
"""


def _kimi_classify(prompt: str) -> str | None:
    adapter = get_kimi_adapter()
    if adapter is None:
        return None
    try:  # pragma: no cover - depends on llama.cpp runtime
        return adapter.generate(
            prompt,
            max_tokens=HALLUCINATION_MAX_TOKENS,
            temperature=HALLUCINATION_TEMPERATURE,
            stop=["\n"],
        ).strip()
    except Exception as exc:
        LOGGER.debug("Kimi-K2 hallucination check failed: %s", exc)
        return None


def _llm_supports(source: str, heading: str) -> bool | None:
    prompt = PROMPT.format(
        source=shorten(source.replace("\n", " "), width=800, placeholder=" ..."),
        heading=heading,
    )
    response = _kimi_classify(prompt)
    if response is None:
        return None
    response = response.lower()
    if response.startswith("yes"):
        return True
    if response.startswith("no"):
        return False
    return None


def _llm_claim_supported(source: str, latex: str) -> bool | None:
    prompt = (
        "Compare the source text with the generated LaTeX. Respond with SUPPORTED if the LaTeX statements are grounded "
        "in the source, otherwise respond UNSUPPORTED.\n\n"
        f"Source:\n{shorten(source.replace(os.linesep, ' '), width=900, placeholder=' ...')}\n\n"
        f"LaTeX:\n{shorten(latex.replace(os.linesep, ' '), width=900, placeholder=' ...')}\n"
    )
    response = _kimi_classify(prompt)
    if response is None:
        return None
    response = response.lower()
    if response.startswith("supported"):
        return True
    if response.startswith("unsupported") or response.startswith("no"):
        return False
    return None


def check_section_headers(
    plan_path: Path,
    chunks_path: Path,
    output_path: Path,
    snippets_path: Path | None = None,
) -> Dict[str, object]:
    plan = common.load_plan(plan_path)
    chunks = {chunk.chunk_id: chunk for chunk in common.load_chunks(chunks_path)}
    snippets = (
        {snippet.chunk_id: snippet.latex for snippet in common.load_snippets(snippets_path)}
        if snippets_path and snippets_path.exists()
        else {}
    )
    report: Dict[str, object] = {"flagged": [], "total": 0, "claim_flags": []}
    for block in plan:
        if block.block_type != "section":
            continue
        chunk = chunks.get(block.chunk_id)
        if not chunk:
            continue
        report["total"] += 1
        heading = block.label or ""
        source = chunk.text or ""
        llm_result = _llm_supports(source, heading)
        if llm_result is None:
            if heading.lower() in source.lower():
                continue
            flagged = True
        else:
            flagged = not llm_result
        if flagged:
            report["flagged"].append(
                {
                    "chunk_id": block.chunk_id,
                    "heading": heading,
                }
            )
        snippet = snippets.get(block.chunk_id)
        if snippet:
            claim_supported = _llm_claim_supported(chunk.text, snippet)
            if claim_supported is False:
                report["claim_flags"].append(
                    {
                        "chunk_id": block.chunk_id,
                        "heading": heading,
                    }
                )
    report["flagged_count"] = len(report["flagged"])
    report["claim_flag_count"] = len(report["claim_flags"])
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


__all__ = ["check_section_headers"]
