"""Helpers that adapt release pipeline components for the backend demo."""
from __future__ import annotations

from typing import Dict, Iterable, Iterator, List

from release.core import common
from release.pipeline import planner

_SAMPLE_CHUNKS: List[common.Chunk] = [
    common.Chunk(
        chunk_id="chunk-intro",
        page=1,
        text="Introduction",
        metadata={"region_type": "heading", "header_level": 1},
    ),
    common.Chunk(
        chunk_id="chunk-overview",
        page=1,
        text="This document demonstrates the streaming planner backed by LaTeXify's release modules.",
        metadata={"region_type": "paragraph"},
    ),
    common.Chunk(
        chunk_id="chunk-equation",
        page=2,
        text=r"E = mc^2 describes the energy/mass equivalence.",
        metadata={"region_type": "formula"},
    ),
    common.Chunk(
        chunk_id="chunk-table",
        page=2,
        text="Table 1 summarizes the metrics across vision branches.",
        metadata={"region_type": "table"},
    ),
]


def demo_chunks() -> Iterable[common.Chunk]:
    return list(_SAMPLE_CHUNKS)


def build_demo_plan() -> Dict[str, object]:
    plan = planner.build_master_plan(demo_chunks(), document_title="LaTeXify Demo")
    return plan.model_dump()


def plan_jobs(plan_payload: Dict[str, object]) -> Iterator[Dict[str, str]]:
    for section in plan_payload.get("sections", []):
        for content in section.get("content", []):
            label = content.get("type", "paragraph").title()
            summary = content.get("summary") or ""
            yield {
                "agent": f"{label}Agent",
                "id": content.get("chunk_id", content.get("item_id", "chunk")),
                "content": summary or label,
            }


PIPELINE_STAGES = [
    ("ingestion", "Reading PDF"),
    ("planner", "Structuring document"),
    ("vision", "Vision branches"),
    ("synthesis", "Generating LaTeX"),
    ("quality", "Quality gating"),
]
