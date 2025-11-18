"""Active-learning queue builder for pipeline runs."""
from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

from ..core import common

DOCUMENT_CHUNK_ID = "__document__"
DEFAULT_LIMIT = int(os.environ.get("LATEXIFY_ACTIVE_LEARNING_LIMIT", "64"))
REASON_PRIORITY: Dict[str, int] = {
    "missing_snippet": 0,
    "validation_error": 0,
    "visual_regression": 1,
    "hallucination": 2,
    "low_quality": 3,
    "lint_warning": 4,
    "low_reward": 5,
}
DEFAULT_PRIORITY = 10


@dataclass
class ActiveLearningSummary:
    summary: Dict[str, object]
    summary_path: Path
    queue_path: Path


def _copy_dict(payload: Dict[str, object] | None) -> Dict[str, object]:
    if not payload:
        return {}
    return dict(payload)


def _priority(reasons: Iterable[str]) -> int:
    values = [REASON_PRIORITY.get(reason, DEFAULT_PRIORITY) for reason in reasons]
    if not values:
        return DEFAULT_PRIORITY
    return min(values)


def build_active_learning_queue(
    run_id: str,
    chunks_path: Path,
    plan_path: Path,
    snippets_path: Path,
    output_dir: Path,
    *,
    quality_report: Dict[str, object] | None = None,
    hallucination_report: Dict[str, object] | None = None,
    gaps_report: Dict[str, object] | None = None,
    visual_report: Dict[str, object] | None = None,
    reward_report: Dict[str, object] | None = None,
    validation_report: Dict[str, object] | None = None,
    lint_report: Dict[str, object] | None = None,
    limit: int | None = None,
) -> ActiveLearningSummary:
    """Collect low-confidence snippets and persist them for human review."""

    chunk_map = {chunk.chunk_id: chunk for chunk in common.load_chunks(chunks_path)} if chunks_path.exists() else {}
    plan_map = {block.chunk_id: block for block in common.load_plan(plan_path)} if plan_path.exists() else {}
    snippet_map = {snippet.chunk_id: snippet for snippet in common.load_snippets(snippets_path)} if snippets_path.exists() else {}
    queue: Dict[str, Dict[str, object]] = {}

    def ensure_record(chunk_id: str) -> Dict[str, object]:
        record = queue.get(chunk_id)
        if record is not None:
            return record
        chunk = chunk_map.get(chunk_id)
        block = plan_map.get(chunk_id)
        snippet = snippet_map.get(chunk_id)
        metadata: Dict[str, object] = {}
        if block:
            metadata["block_id"] = block.block_id
            metadata["block_label"] = block.label
            metadata["block_metadata"] = _copy_dict(block.metadata)
            if block.images:
                metadata["block_images"] = list(block.images)
        if chunk:
            metadata["chunk_metadata"] = _copy_dict(chunk.metadata)
            metadata["chunk_images"] = list(chunk.images)
            metadata["chunk_page"] = chunk.page
        if snippet and snippet.notes:
            metadata["snippet_notes"] = _copy_dict(snippet.notes)
        if chunk_id == DOCUMENT_CHUNK_ID:
            metadata.setdefault("description", "Document-level issue")
        record = {
            "run_id": run_id,
            "chunk_id": chunk_id,
            "page": chunk.page if chunk else metadata.get("chunk_page"),
            "block_type": block.block_type if block else ("document" if chunk_id == DOCUMENT_CHUNK_ID else None),
            "label": block.label if block else ("Document" if chunk_id == DOCUMENT_CHUNK_ID else ""),
            "source_text": chunk.text if chunk else "",
            "generated_latex": snippet.latex if snippet else "",
            "reasons": [],
            "signals": {},
            "metadata": metadata,
        }
        queue[chunk_id] = record
        return record

    def add_reason(chunk_id: str, reason: str, signal: object | None = None) -> None:
        record = ensure_record(chunk_id)
        if reason not in record["reasons"]:
            record["reasons"].append(reason)
        if signal is not None:
            record["signals"][reason] = signal

    weak_sections = (quality_report or {}).get("weak_sections") or []
    for chunk_id in weak_sections:
        add_reason(chunk_id, "low_quality", {"aggregate": (quality_report or {}).get("aggregate")})

    hallucinated = (hallucination_report or {}).get("flagged") or []
    for payload in hallucinated:
        chunk_id = payload.get("chunk_id")
        if chunk_id:
            add_reason(chunk_id, "hallucination", payload)

    missing_chunk_ids = (gaps_report or {}).get("missing_chunk_ids") or []
    for chunk_id in missing_chunk_ids:
        add_reason(chunk_id, "missing_snippet", {"status": "missing"})

    visual_records = []
    if visual_report and visual_report.get("available"):
        visual_records = [rec for rec in visual_report.get("records", []) if rec.get("status") != "ok"]
    flagged_pages = {int(rec.get("page")) for rec in visual_records if rec.get("page")}
    if flagged_pages:
        for chunk in chunk_map.values():
            if chunk.page in flagged_pages:
                add_reason(chunk.chunk_id, "visual_regression", {"page": chunk.page})

    validation_errors = (validation_report or {}).get("errors") or []
    if validation_errors:
        add_reason(DOCUMENT_CHUNK_ID, "validation_error", validation_errors)

    lint_issues = (lint_report or {}).get("issues") or []
    if lint_issues:
        add_reason(DOCUMENT_CHUNK_ID, "lint_warning", lint_issues[:5])

    reward_value = (reward_report or {}).get("reward")
    if reward_value is not None and reward_value < 0:
        add_reason(DOCUMENT_CHUNK_ID, "low_reward", reward_value)

    limit_value = DEFAULT_LIMIT if limit is None else max(1, limit)
    records: List[Dict[str, object]] = sorted(
        queue.values(),
        key=lambda rec: (
            _priority(rec.get("reasons", [])),
            rec.get("page") or 0,
            rec.get("chunk_id", ""),
        ),
    )
    selected = records[:limit_value]

    reason_counts: Counter[str] = Counter()
    for record in selected:
        for reason in record.get("reasons", []):
            reason_counts[reason] += 1

    queue_path = output_dir / "active_learning_queue.jsonl"
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    with queue_path.open("w", encoding="utf-8") as handle:
        for record in selected:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary_path = output_dir / "active_learning_summary.json"
    summary_payload: Dict[str, object] = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "total_candidates": len(selected),
        "reason_counts": dict(reason_counts),
        "quality_score": (quality_report or {}).get("aggregate"),
        "reward": reward_value,
        "visual_flagged_pages": len(flagged_pages),
        "missing_chunks": len(missing_chunk_ids),
        "limit": limit_value,
        "summary_file": str(summary_path),
        "queue_file": str(queue_path),
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    return ActiveLearningSummary(summary=summary_payload, summary_path=summary_path, queue_path=queue_path)


__all__ = ["build_active_learning_queue", "DOCUMENT_CHUNK_ID", "ActiveLearningSummary"]
