#!/usr/bin/env python3
"""Validate planner output and associated layout metadata."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

# Document classes that require the standard LiX frontmatter boilerplate.
LIX_DOC_CLASSES = {
    "lix",
    "lix_article",
    "lix_textbook",
    "textbook",
    "novella",
    "newspaper",
    "contract",
}

# Content types that are considered figures and require an asset.
FIGURE_CONTENT_TYPES = {"figure", "table", "diagram", "chart"}


class ValidationError(Exception):
    """Raised when the plan fails validation."""


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationError(f"Failed to parse JSON from {path}: {exc}") from exc


def _load_jsonl(path: Path) -> List[Any]:
    rows: List[Any] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValidationError(
                    f"Failed to parse JSONL record at line {idx} in {path}: {exc}"
                ) from exc
    return rows


def _load_layout(path: Path | None) -> List[Dict[str, Any]]:
    if not path:
        return []
    if not path.exists():
        raise ValidationError(f"Layout file not found: {path}")
    if path.suffix.lower() == ".jsonl":
        raw = _load_jsonl(path)
    else:
        raw = _load_json(path)
    blocks: List[Dict[str, Any]] = []

    def _visit(obj: Any) -> None:
        if isinstance(obj, dict):
            if {"id", "type"}.issubset(obj.keys()):
                rec = {"id": str(obj["id"]), "type": str(obj["type"]).lower()}
                asset = obj.get("asset_path") or obj.get("image_path") or obj.get("path")
                if isinstance(asset, str) and asset.strip():
                    rec["asset_path"] = asset.strip()
                blocks.append(rec)
            for value in obj.values():
                _visit(value)
        elif isinstance(obj, list):
            for item in obj:
                _visit(item)

    _visit(raw)
    return blocks


def _validate_frontmatter(tasks: List[Dict[str, Any]], doc_class: str) -> None:
    if doc_class not in LIX_DOC_CLASSES:
        return
    required_sequence: Tuple[str, ...] = ("preamble", "titlepage")
    if len(tasks) < len(required_sequence):
        raise ValidationError(
            "LiX-class plans must start with PREAMBLE and TITLE tasks; found fewer tasks."
        )
    for expected_type, task in zip(required_sequence, tasks[: len(required_sequence)]):
        task_type = str(task.get("type", "")).lower()
        if task_type != expected_type:
            raise ValidationError(
                "LiX-class plans must begin with PREAMBLE and TITLE tasks; "
                f"found '{task_type or 'unknown'}' at order {task.get('order')}"
            )


def _validate_task_order(tasks: Iterable[Dict[str, Any]]) -> None:
    seen_ids: set[str] = set()
    previous_order: int | None = None
    for task in tasks:
        tid = str(task.get("id", "")).strip()
        if not tid:
            raise ValidationError("Every task must have a non-empty 'id'.")
        if tid in seen_ids:
            raise ValidationError(f"Duplicate task id detected: {tid}")
        seen_ids.add(tid)
        try:
            order = int(task.get("order"))
        except Exception as exc:  # pragma: no cover - defensive
            raise ValidationError(f"Task '{tid}' is missing a valid integer order.") from exc
        if previous_order is not None and order <= previous_order:
            raise ValidationError(
                "Task order must be strictly increasing: "
                f"{previous_order} -> {order} for task '{tid}'"
            )
        previous_order = order


def _validate_assets(
    plan_path: Path,
    tasks: Iterable[Dict[str, Any]],
    assets_dir: Path,
    layout_blocks: List[Dict[str, Any]],
) -> None:
    assets_dir = assets_dir.resolve()
    layout_index = {
        (rec.get("id"), rec.get("asset_path"))
        for rec in layout_blocks
        if rec.get("asset_path")
    }

    for task in tasks:
        content_type = str(task.get("content_type", "")).lower()
        asset_path = task.get("asset_path")
        if not asset_path:
            if content_type in FIGURE_CONTENT_TYPES:
                raise ValidationError(
                    f"Task '{task.get('id')}' ({content_type}) is missing an asset_path."
                )
            continue

        candidate = Path(asset_path)
        if not candidate.is_absolute():
            candidate = (plan_path.parent / candidate).resolve()
            if not candidate.exists():
                candidate = (assets_dir / Path(asset_path).name).resolve()
        if not candidate.exists():
            raise ValidationError(
                f"Referenced asset for task '{task.get('id')}' not found: {asset_path}"
            )

        if layout_index and (str(task.get("id")), asset_path) not in layout_index:
            # Allow relaxed matching by filename if full path mismatch.
            names = {Path(p or "").name for _, p in layout_index}
            if Path(asset_path).name not in names:
                raise ValidationError(
                    f"Task '{task.get('id')}' asset '{asset_path}' not mentioned in layout metadata."
                )


def validate_plan(plan_path: Path, layout_path: Path | None, assets_dir: Path) -> Dict[str, Any]:
    if not plan_path.exists():
        raise ValidationError(f"Plan file not found: {plan_path}")
    plan = _load_json(plan_path)
    if not isinstance(plan, dict):
        raise ValidationError("Plan JSON must be an object at the top level.")

    doc_class = str(plan.get("doc_class", "")).strip()
    if not doc_class:
        raise ValidationError("Plan is missing 'doc_class'.")

    tasks = plan.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValidationError("Plan must contain a non-empty list of tasks.")

    _validate_task_order(tasks)
    _validate_frontmatter(tasks, doc_class.lower())

    layout_blocks = _load_layout(layout_path)
    _validate_assets(plan_path, tasks, assets_dir, layout_blocks)

    return {
        "doc_class": doc_class,
        "task_count": len(tasks),
        "layout_blocks": len(layout_blocks),
    }


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Sanity-check plan.json against layout metadata.")
    ap.add_argument("--plan", type=Path, default=Path("build/plan.json"))
    ap.add_argument(
        "--layout",
        type=Path,
        default=None,
        help="Optional layout JSON/JSONL produced by OCR/layout analysis.",
    )
    ap.add_argument(
        "--assets-dir",
        type=Path,
        default=Path("build/assets"),
        help="Directory where extracted assets should live.",
    )
    args = ap.parse_args(argv)

    try:
        summary = validate_plan(args.plan, args.layout, args.assets_dir)
    except ValidationError as exc:
        print(f"[validate_plan] ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[validate_plan] ERROR: unexpected failure: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "status": "ok",
                "doc_class": summary["doc_class"],
                "tasks": summary["task_count"],
                "layout_records": summary["layout_blocks"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
