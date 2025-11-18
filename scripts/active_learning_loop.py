#!/usr/bin/env python3
"""Aggregate active-learning candidates from build runs into a managed queue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCUMENT_CHUNK_ID = "__document__"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=REPO_ROOT / "build" / "runs",
        help="Directory containing run artifacts (default: build/runs).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "training_data" / "active_learning" / "queue.jsonl",
        help="Destination JSONL file for the aggregated queue.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Optional metadata JSON path (defaults to <output>.meta.json).",
    )
    parser.add_argument("--limit", type=int, default=100, help="Maximum number of candidates to emit.")
    parser.add_argument(
        "--min-reasons",
        type=int,
        default=1,
        help="Require at least this many reasons per record (default: 1).",
    )
    parser.add_argument(
        "--reason",
        action="append",
        dest="reasons",
        help="Filter to specific reasons (may be repeated).",
    )
    parser.add_argument(
        "--include-document",
        action="store_true",
        help="Include document-level pseudo records (chunk_id='__document__').",
    )
    return parser.parse_args()


def _load_jsonl(path: Path) -> Iterator[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _load_summary(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _iter_run_records(runs_root: Path) -> Iterator[Tuple[Dict[str, object], Dict[str, object], Path]]:
    run_dirs = sorted([entry for entry in runs_root.iterdir() if entry.is_dir()]) if runs_root.exists() else []
    for run_dir in run_dirs:
        reports_dir = run_dir / "reports"
        queue_path = reports_dir / "active_learning_queue.jsonl"
        if not queue_path.exists():
            continue
        summary = _load_summary(reports_dir / "active_learning_summary.json")
        for record in _load_jsonl(queue_path):
            yield record, summary, run_dir


def _sort_key(record: Dict[str, object]) -> Tuple[str, str, str]:
    timestamp = str(record.get("generated_at") or "")
    return (timestamp, str(record.get("run_id") or ""), str(record.get("chunk_id") or ""))


def collect_candidates(args: argparse.Namespace) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    selected: List[Dict[str, object]] = []
    records: List[Dict[str, object]] = []
    for record, summary, run_dir in _iter_run_records(args.runs_root):
        enriched = record.copy()
        enriched.setdefault("run_id", summary.get("run_id") or run_dir.name)
        enriched.setdefault("source_run", run_dir.name)
        enriched.setdefault("generated_at", summary.get("generated_at"))
        records.append(enriched)
    if not records:
        return [], {"runs_scanned": 0, "records_written": 0}
    desired_reasons = set(reason.strip() for reason in (args.reasons or []) if reason and reason.strip())
    include_document = args.include_document
    records.sort(key=_sort_key, reverse=True)
    seen_keys = set()
    limit_value = max(1, args.limit)
    for record in records:
        chunk_id = str(record.get("chunk_id") or "")
        if not include_document and chunk_id == DOCUMENT_CHUNK_ID:
            continue
        reasons = record.get("reasons") or []
        if len(reasons) < args.min_reasons:
            continue
        if desired_reasons and not desired_reasons.intersection(set(reasons)):
            continue
        dedupe_key = (record.get("run_id"), chunk_id)
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        selected.append(record)
        if len(selected) >= limit_value:
            break
    meta = {
        "runs_scanned": len({rec.get("source_run") for rec in records}),
        "records_written": len(selected),
        "limit": limit_value,
        "filters": {
            "min_reasons": args.min_reasons,
            "reasons": sorted(desired_reasons) if desired_reasons else [],
            "include_document": include_document,
        },
    }
    return selected, meta


def main() -> None:
    args = parse_args()
    records, meta = collect_candidates(args)
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    metadata_path = args.metadata or output_path.with_suffix(output_path.suffix + ".meta.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_payload = meta | {"output": str(output_path), "runs_root": str(args.runs_root)}
    metadata_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")
    print(f"[active-learning] wrote {meta['records_written']} record(s) to {output_path}")


if __name__ == "__main__":
    main()
