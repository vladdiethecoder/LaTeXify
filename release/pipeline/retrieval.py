"""Multimodal retrieval and embedding helpers for the release pipeline."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List

from PIL import Image, ImageStat

from ..core import common

TOKEN_RE = re.compile(r"\w+")


def _token_stats(text: str) -> Dict[str, float]:
    tokens = TOKEN_RE.findall(text.lower())
    unique = set(tokens)
    length = len(text)
    return {
        "token_count": len(tokens),
        "unique_ratio": (len(unique) / len(tokens)) if tokens else 0.0,
        "avg_token_len": (sum(len(t) for t in tokens) / len(tokens)) if tokens else 0.0,
        "char_len": length,
    }


def _image_signature(image_path: str | None) -> Dict[str, float] | None:
    if not image_path:
        return None
    path = Path(image_path)
    if not path.exists():
        return None
    with Image.open(path) as image:
        gray = image.convert("L")
        stats = ImageStat.Stat(gray)
        mean = stats.mean[0]
        std = stats.stddev[0]
        return {
            "mean_intensity": mean,
            "std_intensity": std,
            "width": image.width,
            "height": image.height,
            "aspect_ratio": image.width / image.height if image.height else 0,
            "color_histogram": _histogram_signature(image),
        }


def _histogram_signature(image: Image.Image) -> List[float]:
    resized = image.resize((32, 32)).convert("RGB")
    hist = resized.histogram()
    total = sum(hist) or 1
    return [round(value / total, 6) for value in hist]


def _modalities(chunk: common.Chunk) -> Dict[str, bool]:
    meta = chunk.metadata or {}
    return {
        "text": True,
        "figure": bool(meta.get("region_type") == "figure" or chunk.images),
        "table": bool(meta.get("region_type") == "table"),
        "formula": bool(meta.get("formula_detected")),
    }


def _weight_for_region(region: str | None) -> float:
    if region == "table":
        return 1.35
    if region == "figure":
        return 1.25
    if region in {"equation", "formula"}:
        return 1.15
    if region == "heading":
        return 0.95
    return 1.0


def build_index(chunks_path: Path, plan_path: Path, output_path: Path) -> Path:
    chunks = common.load_chunks(chunks_path)
    plan = {block.chunk_id: block for block in common.load_plan(plan_path)}
    entries: List[Dict[str, object]] = []
    for chunk in chunks:
        meta = chunk.metadata or {}
        plan_meta = plan.get(chunk.chunk_id).metadata if plan.get(chunk.chunk_id) else {}
        features = _token_stats(chunk.text)
        if meta.get("table_signature"):
            features.update(
                {
                    "table_rows": meta["table_signature"].get("rows", 0),
                    "table_cols": meta["table_signature"].get("columns", 0),
                    "table_density": (meta["table_signature"].get("rows", 1) * meta["table_signature"].get("columns", 1)),
                }
            )
        if meta.get("list_depth"):
            features["list_depth"] = meta["list_depth"]
        features["header_level"] = meta.get("header_level", 0)
        region = meta.get("region_type", "text")
        digest = hashlib.sha1(region.encode("utf-8")).hexdigest()
        features["region_type_hash"] = int(digest[:6], 16)
        image_signature = _image_signature(meta.get("page_image"))
        weight = _weight_for_region(region)
        entry = {
            "chunk_id": chunk.chunk_id,
            "modalities": _modalities(chunk),
            "features": features,
            "image_signature": image_signature,
            "plan_block_type": plan.get(chunk.chunk_id).block_type if plan.get(chunk.chunk_id) else None,
            "plan_metadata": plan_meta,
            "weight": weight,
        }
        entries.append(entry)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    return output_path


__all__ = ["build_index"]
