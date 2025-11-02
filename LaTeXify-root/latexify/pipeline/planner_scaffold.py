"""
Lightweight planner for constructing a document plan from a list of question
identifiers, optional layout analysis and asset manifest information.

The planner produces a JSON-compatible ``plan`` dict with the following
structure::

    {
        "doc_class": <document class>,
        "frontmatter": {"title": ..., "author": ..., "course": ..., "date": ...},
        "tasks": [
            {"id": "PREAMBLE", "type": "preamble", "order": 0, "content_type": "frontmatter"},
            {"id": "TITLE",    "type": "titlepage", "order": 1, "content_type": "frontmatter"},
            {"id": "Q1", ...},
            ...
        ]
    }

Each question identifier is assigned an increasing ``order`` index.  Layout
information can be provided via a JSON or JSONL file to indicate block
types (e.g. Figure, Table) and page indices.  An optional asset manifest
maps asset identifiers or block identifiers to actual filenames on disk.  If
an asset is found for a given block and is deemed "visual", the planner
will emit a ``figure`` task instead of a generic ``question``.  Missing
visual assets result in a ``figure_placeholder`` task to signal that a
figure is expected but has not yet been resolved.

The functions defined here are pure and side-effect free.  A small CLI
wrapper lives in ``scripts/planner_scaffold.py`` and delegates to these
functions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence


@dataclass
class LayoutBlock:
    """Represents a block detected during layout analysis.

    Attributes:
        block_id: Identifier for the block, typically corresponding to a
            question id.
        content_type: Semantic type of the block (e.g. ``Figure``, ``Table``,
            ``Math``).  Used to decide whether a visual asset should be
            included.
        page_index: Zero-based page index where the block appears.  Used as a
            fallback when matching assets by location.
        asset_id: Identifier referencing an entry in the asset manifest.
    """

    block_id: str
    content_type: str | None = None
    page_index: int | None = None
    asset_id: str | None = None


@dataclass
class AssetInfo:
    """Describes a single asset exported from a PDF.

    Attributes:
        asset_path: Relative path to the asset file on disk.
        asset_type: Semantic type of the asset (e.g. ``Figure``, ``Table``).
        page_index: Zero-based page index of the asset.  Used as a fallback
            when matching by location.
        block_id: Identifier of the block this asset corresponds to, if any.
        asset_id: Explicit identifier of the asset; falls back to the stem of
            the filename if not provided.
    """

    asset_path: str
    asset_type: str | None = None
    page_index: int | None = None
    block_id: str | None = None
    asset_id: str | None = None


class AssetLookup:
    """Provides efficient lookup of assets by id, block id or page index."""

    def __init__(self) -> None:
        self.by_id: Dict[str, AssetInfo] = {}
        self.by_page: Dict[int, List[AssetInfo]] = {}
        self.ordered: List[AssetInfo] = []

    def add(self, entry: AssetInfo) -> None:
        """Register a new asset into the lookup tables."""
        self.ordered.append(entry)
        # Keys can come from asset_id or block_id (strings only)
        for key in filter(None, (entry.asset_id, entry.block_id)):
            self.by_id.setdefault(str(key), entry)
        if entry.page_index is not None:
            try:
                idx = int(entry.page_index)
            except (TypeError, ValueError):
                return
            self.by_page.setdefault(idx, []).append(entry)

    def match_for_block(self, block: LayoutBlock, used_paths: set[str]) -> AssetInfo | None:
        """Find an unused asset matching the given block.

        Matching proceeds by checking the asset_id and block_id first.  If
        neither yields a usable asset, the page_index is used as a fallback.
        Assets already used in previous tasks are skipped.
        """
        for key in filter(None, (block.asset_id, block.block_id)):
            info = self.by_id.get(str(key))
            if info and info.asset_path not in used_paths:
                return info
        if block.page_index is not None:
            try:
                idx = int(block.page_index)
            except (TypeError, ValueError):
                idx = None
            if idx is not None:
                for info in self.by_page.get(idx, []):
                    if info.asset_path not in used_paths:
                        return info
        return None

    @property
    def ordered_paths(self) -> List[str]:
        """Return asset paths in the order they were added."""
        return [entry.asset_path for entry in self.ordered]


def _split_list(csv: str | None) -> List[str]:
    """Split a comma-separated string into a list of stripped tokens."""
    if not csv:
        return []
    return [s.strip() for s in csv.split(",") if s.strip()]


def _looks_like_figure(value: str | None) -> bool:
    if not value:
        return False
    value = value.lower()
    return any(token in value for token in (
        "figure",
        "image",
        "picture",
        "graphic",
        "diagram",
        "photo",
        "chart",
    ))


def _looks_like_table(value: str | None) -> bool:
    if not value:
        return False
    value = value.lower()
    return any(token in value for token in ("table", "spreadsheet", "tabular"))


def _looks_like_math(value: str | None) -> bool:
    if not value:
        return False
    value = value.lower()
    return any(token in value for token in ("formula", "equation", "math", "expression"))


def _asset_is_visual(*candidates: str | None) -> bool:
    """Return True if any candidate string describes a visual asset."""
    for candidate in candidates:
        if not candidate:
            continue
        if (
            _looks_like_figure(candidate)
            or _looks_like_table(candidate)
            or _looks_like_math(candidate)
        ):
            return True
    return False


def _coerce_str(value) -> str | None:
    """Attempt to coerce any object to a string, returning None on failure."""
    if value is None:
        return None
    try:
        return str(value)
    except Exception:
        return None


def _coerce_int(value) -> int | None:
    """Attempt to coerce any object to an integer, returning None on failure."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _emit_plan(
    doc_class: str,
    *,
    title: str = "Untitled Document",
    author: str = "Generated by LaTeXify",
    course: str = "",
    date: str = r"\today",
    questions: Sequence[str] | None = None,
    layout_blocks: Dict[str, LayoutBlock] | None = None,
    assets: AssetLookup | None = None,
) -> dict:
    """Construct a document plan given optional layout and asset information."""
    layout_blocks = layout_blocks or {}
    assets = assets or AssetLookup()

    tasks: List[Dict[str, object]] = [
        {"id": "PREAMBLE", "type": "preamble", "order": 0, "content_type": "frontmatter"},
        {"id": "TITLE", "type": "titlepage", "order": 1, "content_type": "frontmatter"},
    ]
    used_assets: set[str] = set()
    # Enumerate questions starting from 2 (after preamble and title)
    for i, q in enumerate(questions or (), start=2):
        block = layout_blocks.get(q, LayoutBlock(block_id=q))
        entry: Dict[str, object] = {
            "id": q,
            "type": "question",
            "title": q.replace("_", " "),
            "order": i,
        }
        # Carry over any explicit content type from layout analysis
        if block.content_type:
            entry["content_type"] = block.content_type
        # Attempt to match an asset for the block
        asset_info = assets.match_for_block(block, used_assets)
        if asset_info:
            semantic_type = asset_info.asset_type or block.content_type
            # If the asset or block implies a visual type, emit a figure
            if _asset_is_visual(semantic_type, block.content_type):
                entry["type"] = "figure"
                entry["asset_path"] = asset_info.asset_path
                entry["asset_source_type"] = semantic_type
                if asset_info.asset_id:
                    entry["asset_id"] = asset_info.asset_id
                if asset_info.page_index is not None:
                    entry["asset_page_index"] = int(asset_info.page_index)
                used_assets.add(asset_info.asset_path)
            else:
                # Asset exists but not a visual block; leave as regular content
                entry.setdefault("type", "question")
        else:
            # No asset found; if block indicates a visual type, demote to placeholder
            if _looks_like_figure(block.content_type):
                entry["type"] = "figure_placeholder"
                if block.content_type:
                    entry["asset_source_type"] = block.content_type
        tasks.append(entry)
    plan = {
        "doc_class": doc_class,
        "frontmatter": {
            "title": title,
            "author": author,
            "course": course,
            "date": date,
        },
        "tasks": tasks,
    }
    return plan


def validate_plan(plan: dict) -> None:
    """Ensure that a plan has unique task IDs and strictly increasing order."""
    seen_ids: set[str] = set()
    previous_order: int | None = None
    for task in plan.get("tasks", []):
        task_id = task.get("id")
        if task_id in seen_ids:
            raise SystemExit(f"Duplicate task id detected: {task_id}")
        seen_ids.add(task_id)
        order = task.get("order")
        if previous_order is not None and order <= previous_order:
            raise SystemExit(
                "Task order must be strictly increasing: "
                f"{previous_order} -> {order} for task {task_id}"
            )
        previous_order = order


def _load_layout_blocks(path: Path | None) -> Dict[str, LayoutBlock]:
    """Parse a layout JSON or JSONL file into a mapping of block id → LayoutBlock."""
    if not path or not path.exists():
        return {}
    try:
        if path.suffix.lower() == ".jsonl":
            data = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    blocks: Dict[str, LayoutBlock] = {}

    def _ensure_block(identifier) -> LayoutBlock | None:
        ident = _coerce_str(identifier)
        if ident is None:
            return None
        if ident not in blocks:
            blocks[ident] = LayoutBlock(block_id=ident)
        return blocks[ident]

    def _update_block(obj: Dict) -> None:
        candidate_ids = [
            obj.get("id"),
            obj.get("block_id"),
            obj.get("task_id"),
            obj.get("question_id"),
            obj.get("bundle_id"),
        ]
        block: LayoutBlock | None = None
        for cid in candidate_ids:
            block = _ensure_block(cid)
            if block is not None:
                break
        if block is None:
            return
        ctype = _coerce_str(obj.get("type") or obj.get("block_type") or obj.get("category"))
        if ctype:
            block.content_type = ctype
        page = _coerce_int(obj.get("page_index") or obj.get("page") or obj.get("page_number"))
        if page is not None:
            block.page_index = page
        asset_ref = _coerce_str(
            obj.get("asset_id")
            or obj.get("image_id")
            or obj.get("asset_ref")
            or obj.get("asset")
        )
        if asset_ref:
            block.asset_id = block.asset_id or asset_ref

    def _walk(obj):
        if isinstance(obj, dict):
            _update_block(obj)
            for value in obj.values():
                _walk(value)
        elif isinstance(obj, list):
            for item in obj:
                _walk(item)

    _walk(data)
    return blocks


def _load_asset_manifest(path: Path | None) -> AssetLookup:
    """Load an asset manifest JSON into an ``AssetLookup``.

    The manifest can either be a list of entries or a dict containing an
    ``entries`` or ``assets`` list.  Each entry is expected to be a mapping
    containing at least the relative path or filename of the asset.  Optional
    keys include ``asset_type``, ``type`` or ``block_type``, ``page_index``
    and identifiers like ``block_id`` or ``asset_id``.  Unknown fields are
    ignored.
    """
    lookup = AssetLookup()
    if not path or not path.exists():
        return lookup
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return lookup
    entries: List[dict] = []
    if isinstance(data, dict):
        if isinstance(data.get("entries"), list):
            entries = data.get("entries", [])
        elif isinstance(data.get("assets"), list):
            entries = data.get("assets", [])
    elif isinstance(data, list):
        entries = data  # type: ignore[assignment]
    for item in entries:
        if not isinstance(item, dict):
            continue
        rel = (
            item.get("asset_path")
            or item.get("relative_path")
            or item.get("rel_path")
            or item.get("path")
            or item.get("filename")
        )
        rel = _coerce_str(rel)
        if not rel:
            continue
        asset_type = _coerce_str(
            item.get("asset_type")
            or item.get("type")
            or item.get("block_type")
            or item.get("category")
        )
        page_index = _coerce_int(item.get("page_index") or item.get("page") or item.get("page_number"))
        block_id = _coerce_str(
            item.get("block_id")
            or item.get("task_id")
            or item.get("question_id")
            or item.get("bundle_id")
        )
        asset_id = _coerce_str(
            item.get("asset_id")
            or item.get("image_id")
            or item.get("id")
            or item.get("asset_identifier")
        )
        if not asset_id:
            asset_id = Path(rel).stem
        info = AssetInfo(
            asset_path=rel,
            asset_type=asset_type,
            page_index=page_index,
            block_id=block_id,
            asset_id=asset_id,
        )
        lookup.add(info)
    return lookup


__all__ = [
    "LayoutBlock",
    "AssetInfo",
    "AssetLookup",
    "_emit_plan",
    "validate_plan",
    "_load_layout_blocks",
    "_load_asset_manifest",
    "_split_list",
]