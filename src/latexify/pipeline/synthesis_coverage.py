"""Utilities for checking plan/snippet coverage."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from ..core import common


def _expected_chunk_ids(master_plan_path: Path) -> List[str]:
    data = json.loads(master_plan_path.read_text(encoding="utf-8"))
    expected: List[str] = []
    for section in data.get("sections", []):
        for content in section.get("content", []):
            chunk_id = content.get("chunk_id")
            if chunk_id:
                expected.append(chunk_id)
    return expected


def find_gaps(master_plan_path: Path, snippets_path: Path) -> Dict[str, object]:
    report = {
        "expected_snippets": 0,
        "actual_snippets": 0,
        "missing_chunk_ids": [],
    }
    if not master_plan_path.exists() or not snippets_path.exists():
        return report
    expected = _expected_chunk_ids(master_plan_path)
    snippets = common.load_snippets(snippets_path)
    actual_ids = {snippet.chunk_id for snippet in snippets}
    missing = [chunk_id for chunk_id in expected if chunk_id not in actual_ids]
    report["expected_snippets"] = len(expected)
    report["actual_snippets"] = len(snippets)
    report["missing_chunk_ids"] = missing
    return report


__all__ = ["find_gaps"]
