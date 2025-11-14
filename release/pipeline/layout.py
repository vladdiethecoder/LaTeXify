"""Layout planner informed by SCAN/Detect-Order-Construct style heuristics."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable, List, Dict, Set

from ..core import common
from .planner import MasterPlan, load_master_plan

LOGGER = logging.getLogger(__name__)
SECTION_RE = re.compile(r"^(chapter|section|appendix|part|lesson)\b", flags=re.IGNORECASE)
EQUATION_RE = re.compile(r"=|\\sum|\\int|\\frac")


def infer_label(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if len(stripped) <= 72:
            return stripped
    return text.splitlines()[0].strip() if text else "Untitled"


def classify_block(chunk: common.Chunk) -> str:
    metadata = chunk.metadata or {}
    region = metadata.get("region_type")
    header_level = metadata.get("header_level", 0)
    if region == "question" or metadata.get("question_label"):
        return "question"
    if region == "figure":
        return "figure"
    if region == "table":
        return "table"
    if region == "list":
        return "list"
    if metadata.get("formula_detected"):
        return "equation"
    first_line = chunk.text.splitlines()[0].strip() if chunk.text else ""
    uppercase_line = first_line.isupper() and len(first_line.split()) <= 8
    if header_level > 0 or SECTION_RE.match(first_line) or uppercase_line:
        return "section"
    if chunk.images:
        return "figure"
    if EQUATION_RE.search(chunk.text):
        return "equation"
    return "text"


def build_plan(chunks: List[common.Chunk]) -> Iterable[common.PlanBlock]:
    for idx, chunk in enumerate(chunks):
        block_type = classify_block(chunk)
        block_metadata = {
            "region_type": chunk.metadata.get("region_type"),
            "header_level": chunk.metadata.get("header_level", 0),
            "list_depth": chunk.metadata.get("list_depth"),
            "formula_detected": chunk.metadata.get("formula_detected", False),
            "table_signature": chunk.metadata.get("table_signature"),
        }
        yield common.PlanBlock(
            block_id=f"block_{idx:04d}",
            chunk_id=chunk.chunk_id,
            label=infer_label(chunk.text),
            block_type=block_type,
            images=chunk.images,
            metadata=block_metadata,
        )


def _blocks_from_master_plan(master_plan: MasterPlan, chunk_map: Dict[str, common.Chunk]) -> List[common.PlanBlock]:
    blocks: List[common.PlanBlock] = []
    seen: Set[str] = set()

    def append_chunk(chunk_id: str) -> None:
        if not chunk_id or chunk_id in seen:
            return
        chunk = chunk_map.get(chunk_id)
        if not chunk:
            LOGGER.warning("Master plan references missing chunk %s", chunk_id)
            return
        block_type = classify_block(chunk)
        block_metadata = {
            "region_type": chunk.metadata.get("region_type"),
            "header_level": chunk.metadata.get("header_level", 0),
            "list_depth": chunk.metadata.get("list_depth"),
            "formula_detected": chunk.metadata.get("formula_detected", False),
            "table_signature": chunk.metadata.get("table_signature"),
        }
        blocks.append(
            common.PlanBlock(
                block_id=f"block_{len(blocks):04d}",
                chunk_id=chunk.chunk_id,
                label=infer_label(chunk.text),
                block_type=block_type,
                images=chunk.images,
                metadata=block_metadata,
            )
        )
        seen.add(chunk_id)

    for section in master_plan.sections:
        if section.heading_chunk_id:
            append_chunk(section.heading_chunk_id)
        for content in section.content:
            append_chunk(content.chunk_id)
    missing = [chunk.chunk_id for chunk in chunk_map.values() if chunk.chunk_id not in seen]
    for chunk_id in missing:
        append_chunk(chunk_id)
    return blocks


def run_layout(chunks_path: Path, plan_path: Path, master_plan_path: Path | None = None) -> Path:
    chunks = common.load_chunks(chunks_path)
    chunk_map = {chunk.chunk_id: chunk for chunk in chunks}
    if master_plan_path and master_plan_path.exists():
        master_plan = load_master_plan(master_plan_path)
        plan = _blocks_from_master_plan(master_plan, chunk_map)
    else:
        plan = list(build_plan(chunks))
    common.save_plan(plan, plan_path)
    LOGGER.info("Layout planning complete with %s blocks", len(plan))
    return plan_path


__all__ = ["run_layout"]
