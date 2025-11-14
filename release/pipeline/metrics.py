"""Structured metrics for benchmarking each pipeline stage."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

from ..core import common

SECTION_RE = re.compile(r"\\section|\\subsection|\\subsubsection")
TABLE_RE = re.compile(r"\\begin\{table\}")
LIST_RE = re.compile(r"\\begin\{(itemize|enumerate)\}")
EQUATION_RE = re.compile(r"\\begin\{equation\}")


def _ratio(count: int, total: int) -> float:
    return 1.0 if total == 0 else min(1.0, count / total)


def evaluate(
    plan_path: Path,
    tex_path: Path,
    retrieval_path: Path,
    output_path: Path,
    chunks_path: Path,
    validation_path: Path,
) -> Path:
    plan = common.load_plan(plan_path)
    tex = tex_path.read_text(encoding="utf-8") if tex_path.exists() else ""
    retrieval_entries: List[Dict[str, object]] = []
    if retrieval_path.exists():
        retrieval_entries = json.loads(retrieval_path.read_text(encoding="utf-8"))
    chunks = common.load_chunks(chunks_path) if chunks_path.exists() else []
    validation_data = json.loads(validation_path.read_text(encoding="utf-8")) if validation_path.exists() else {}
    sections = [b for b in plan if b.block_type == "section"]
    tables = [b for b in plan if b.block_type == "table"]
    lists = [b for b in plan if b.block_type == "list"]
    equations = [b for b in plan if b.block_type == "equation"]
    figures = [b for b in plan if b.block_type == "figure"]
    metrics = {
        "section_fidelity": _ratio(len(SECTION_RE.findall(tex)), len(sections)),
        "table_fidelity": _ratio(len(TABLE_RE.findall(tex)), len(tables)),
        "list_integrity": _ratio(len(LIST_RE.findall(tex)), len(lists)),
        "equation_fidelity": _ratio(len(EQUATION_RE.findall(tex)), len(equations)),
    }
    missing_figures = 0
    for block in figures:
        linked = any(Path(image).name in tex for image in block.images)
        if not linked:
            missing_figures += 1
    metrics["figure_linkage"] = 1.0 - _ratio(missing_figures, max(1, len(figures)))
    modality_counts = {"text": 0, "figure": 0, "table": 0, "formula": 0}
    for entry in retrieval_entries:
        modalities = entry.get("modalities") or {}
        for modality, present in modalities.items():
            if present and modality in modality_counts:
                modality_counts[modality] += 1
    metrics["modalities"] = modality_counts
    if chunks:
        noise_by_region: Dict[str, Dict[str, float]] = {}
        for chunk in chunks:
            region = chunk.metadata.get("region_type", "text")
            noise_by_region.setdefault(region, {"count": 0, "noise": 0.0})
            rec = noise_by_region[region]
            rec["count"] += 1
            rec["noise"] += chunk.metadata.get("noise_score", 0.0)
        for region, rec in noise_by_region.items():
            rec["avg_noise"] = rec["noise"] / max(1, rec["count"])
            del rec["noise"]
        metrics["ocr_noise_by_region"] = noise_by_region
    section_levels = [block.metadata.get("header_level", 1) for block in plan if block.block_type == "section"]
    if section_levels:
        jumps = sum(1 for a, b in zip(section_levels, section_levels[1:]) if (b - a) > 1)
        metrics["section_tree_integrity"] = 1.0 - _ratio(jumps, max(1, len(section_levels) - 1))
    if retrieval_entries:
        grounded = sum(1 for entry in retrieval_entries if entry.get("weight", 1.0) > 1.0 and entry.get("image_signature"))
        metrics["retrieval_grounding"] = _ratio(grounded, len(retrieval_entries))
    metrics["compile_success"] = bool(validation_data.get("success"))
    metrics["compile_errors"] = validation_data.get("errors", [])
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return output_path


__all__ = ["evaluate"]
