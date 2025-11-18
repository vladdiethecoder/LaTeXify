"""LLM-based hallucination checks for section headers."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from textwrap import shorten
from typing import Dict, List

from ..core import common
from ..models.vllm_client import get_vllm_client

LOGGER = logging.getLogger(__name__)
HALLUCINATION_MODEL = os.environ.get("LATEXIFY_HALLUCINATION_MODEL", "deepseek-ai/DeepSeek-V3")
PROMPT = """Determine if the proposed section heading is supported by the source text.

Source:
{source}

Heading:
{heading}

Respond with YES if the heading clearly appears or is implied by the source, otherwise respond NO.
"""


def _llm_supports(source: str, heading: str) -> bool | None:
    client = get_vllm_client(model=HALLUCINATION_MODEL)
    if client is None:
        return None
    prompt = PROMPT.format(
        source=shorten(source.replace("\n", " "), width=800, placeholder=" ..."),
        heading=heading,
    )
    try:  # pragma: no cover - depends on local LLM
        response = client.generate(prompt, stop=[], max_tokens=64).strip().lower()
    except Exception as exc:
        LOGGER.debug("Hallucination LLM check failed: %s", exc)
        return None
    if response.startswith("yes"):
        return True
    if response.startswith("no"):
        return False
    return None


def _llm_claim_supported(source: str, latex: str) -> bool | None:
    client = get_vllm_client(model=HALLUCINATION_MODEL)
    if client is None:
        return None
    prompt = (
        "Compare the source text with the generated LaTeX. Respond with SUPPORTED if the LaTeX statements are grounded "
        "in the source, otherwise respond UNSUPPORTED.\n\n"
        f"Source:\n{shorten(source.replace(os.linesep, ' '), width=900, placeholder=' ...')}\n\n"
        f"LaTeX:\n{shorten(latex.replace(os.linesep, ' '), width=900, placeholder=' ...')}\n"
    )
    try:  # pragma: no cover - model runtime
        response = client.generate(prompt, max_tokens=96).strip().lower()
    except Exception:
        return None
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
