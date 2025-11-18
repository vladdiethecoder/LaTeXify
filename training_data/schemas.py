"""Shared LayoutJSONL schema validation helpers."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List


def _is_bbox(bbox: Iterable[Any]) -> bool:
    values = list(bbox)
    if len(values) != 4:
        return False
    return all(isinstance(value, (int, float)) for value in values)


def validate_record(record: Dict[str, Any]) -> None:
    """Lightweight schema enforcement for LayoutJSONL rows."""

    required_top = (
        "id",
        "slug",
        "split",
        "page_index",
        "image_path",
        "tokens",
        "blocks",
        "classes",
        "metadata",
    )
    if not isinstance(record, dict):
        raise TypeError("LayoutJSONL row must be a dict.")
    for field in required_top:
        if field not in record:
            raise KeyError(f"LayoutJSONL row missing '{field}'")
    if record["split"] not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split '{record['split']}'")
    if not isinstance(record["tokens"], list):
        raise TypeError("'tokens' must be a list.")
    for token in record["tokens"]:
        if not isinstance(token, dict):
            raise TypeError("Token entries must be dicts.")
        if "text" not in token or "bbox" not in token:
            raise KeyError("Token missing 'text' or 'bbox'.")
        if not _is_bbox(token["bbox"]):
            raise ValueError("Token bbox must be four numeric values.")
    if not isinstance(record["blocks"], list):
        raise TypeError("'blocks' must be a list.")
    for block in record["blocks"]:
        if not isinstance(block, dict):
            raise TypeError("Block entries must be dicts.")
        if "id" not in block or "type" not in block or "bbox" not in block:
            raise KeyError("Block missing 'id', 'type', or 'bbox'.")
        if not _is_bbox(block["bbox"]):
            raise ValueError("Block bbox must be four numeric values.")
        children = block.get("children", [])
        if not isinstance(children, list):
            raise TypeError("'children' must be a list when provided.")
        for child_idx in children:
            if not isinstance(child_idx, int):
                raise TypeError("Child indices must be integers.")

