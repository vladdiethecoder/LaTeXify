"""Build LayoutJSONL splits from raw dataset folders."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Sequence

import pyarrow.parquet as pq
from PIL import Image

from .schemas import validate_record

REPO_ROOT = Path(__file__).resolve().parents[1]


def register_converter(slug: str):
    def decorator(func: Callable[[Path, Path], Dict[str, int]]):
        CONVERTERS[slug] = func
        return func

    return decorator


def _write_jsonl(records: Iterable[dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            validate_record(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def _normalized_bbox(bbox, width: float, height: float) -> List[float]:
    x, y, w, h = bbox
    return [float(x), float(y), float(x + w), float(y + h)]


CONVERTERS: Dict[str, Callable[[Path, Path], Dict[str, int]]] = {}


def _parquet_files(data_dir: Path) -> Dict[str, List[Path]]:
    files_by_split: Dict[str, List[Path]] = defaultdict(list)
    for path in sorted(data_dir.glob("*.parquet")):
        prefix = path.name.split("-")[0]
        split = {"validation": "val", "dev": "val"}.get(prefix, prefix)
        files_by_split[split].append(path)
    return files_by_split


def _iter_parquet_rows(path: Path, batch_size: int = 128) -> Iterator[dict]:
    parquet_file = pq.ParquetFile(path)
    for record_batch in parquet_file.iter_batches(batch_size=batch_size):
        for row in record_batch.to_pylist():
            yield row


def _ensure_image(image_payload: dict, out_dir: Path, stem: str) -> tuple[str, int, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    image_path = out_dir / f"{stem}.jpg"
    bytes_data = image_payload.get("bytes")
    if bytes_data:
        with Image.open(BytesIO(bytes_data)) as image_obj:
            width, height = image_obj.size
            image_obj.save(image_path, format="JPEG")
    elif image_payload.get("path"):
        src = Path(image_payload["path"])
        image_path.write_bytes(src.read_bytes())
        with Image.open(image_path) as image_obj:
            width, height = image_obj.size
    else:
        raise ValueError("Image payload missing data.")
    rel_path = str(image_path.relative_to(REPO_ROOT))
    return rel_path, width, height


@register_converter("publaynet")
def build_publaynet(root: Path, converted_root: Path) -> Dict[str, int]:
    """Convert PubLayNet parquet shards into LayoutJSONL."""

    data_dir = root / "source" / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"PubLayNet parquet directory missing at {data_dir}")
    files_by_split = _parquet_files(data_dir)
    output_base = converted_root / "publaynet"
    stats: Dict[str, int] = {}
    category_map = {1: "text", 2: "title", 3: "list", 4: "table", 5: "figure"}

    for split, files in files_by_split.items():
        image_dir = root / "images" / split
        records = []
        for parquet_file in files:
            for row in _iter_parquet_rows(parquet_file):
                stem = str(row.get("id"))
                image_payload = row.get("image") or {}
                if not image_payload:
                    continue
                try:
                    image_path, width, height = _ensure_image(image_payload, image_dir, stem)
                except Exception:
                    continue
                blocks = []
                for idx, ann in enumerate(row.get("annotations", [])):
                    bbox = ann.get("bbox") or [0, 0, 0, 0]
                    block = {
                        "id": f"b{idx}",
                        "type": category_map.get(ann.get("category_id"), "other"),
                        "bbox": _normalized_bbox(bbox, width, height),
                        "children": [],
                    }
                    blocks.append(block)
                record = {
                    "id": f"publaynet/{split}/{stem}",
                    "slug": "publaynet",
                    "split": split,
                    "page_index": 0,
                    "image_path": image_path,
                    "width": width,
                    "height": height,
                    "tokens": [],
                    "blocks": blocks,
                    "classes": [],
                    "metadata": {"source": "publaynet", "parquet": parquet_file.name},
                }
                records.append(record)
        stats[split] = _write_jsonl(records, output_base / f"{split}.jsonl")
    return stats


@register_converter("doclaynet")
def build_doclaynet(root: Path, converted_root: Path) -> Dict[str, int]:
    """Convert DocLayNet COCO annotations to LayoutJSONL."""

    coco_dir = root / "source" / "DocLayNet_core" / "COCO"
    extra_dir = root / "source" / "DocLayNet_extra" / "JSON"
    if not coco_dir.exists() or not extra_dir.exists():
        raise FileNotFoundError("DocLayNet assets missing; run downloader first.")

    mapping = {"caption": "caption"}
    stats: Dict[str, int] = {}
    for split in ("train", "val", "test"):
        coco_path = coco_dir / f"{split}.json"
        if not coco_path.exists():
            continue
        payload = json.loads(coco_path.read_text())
        annotations = defaultdict(list)
        for ann in payload.get("annotations", []):
            annotations[ann["image_id"]].append(ann)

        records = []
        for image in payload.get("images", []):
            image_id = image["id"]
            file_name = image["file_name"]
            page_id = Path(file_name).stem
            extra_path = extra_dir / f"{page_id}.json"
            if not extra_path.exists():
                continue
            extra = json.loads(extra_path.read_text())
            width = float(image.get("width") or extra.get("metadata", {}).get("coco_width") or 1000)
            height = float(image.get("height") or extra.get("metadata", {}).get("coco_height") or 1000)
            blocks = []
            tokens = []
            for idx, ann in enumerate(annotations.get(image_id, [])):
                label = mapping.get(ann.get("category_id"), ann.get("category", "other"))
                bbox = _normalized_bbox(ann.get("bbox", [0, 0, 0, 0]), width, height)
                blocks.append({"id": f"b{idx}", "type": label, "bbox": bbox, "children": []})
            for cell in extra.get("cells", []):
                bbox = cell.get("bbox")
                if not bbox:
                    continue
                text = cell.get("text") or ""
                if not text.strip():
                    continue
                normalized = _normalized_bbox(bbox, width, height)
                block_id = None
                tokens.append({"text": text.strip(), "bbox": normalized, "block_id": block_id})
            record = {
                "id": f"doclaynet/{split}/{page_id}",
                "slug": "doclaynet",
                "split": split,
                "page_index": 0,
                "image_path": f"DocLayNet_core/images/{split}/{file_name}",
                "width": width,
                "height": height,
                "tokens": tokens,
                "blocks": blocks,
                "classes": [],
                "metadata": {"source": "doclaynet", "image_id": image_id},
            }
            records.append(record)
        stats[split] = _write_jsonl(records, converted_root / "doclaynet" / f"{split}.jsonl")
    return stats


@register_converter("docbank")
def build_docbank(root: Path, converted_root: Path) -> Dict[str, int]:
    """Convert DocBank TSV to LayoutJSONL."""

    source_dir = root / "source"
    txt_root = source_dir / "DocBank_500K_txt"
    if (txt_root / "DocBank_500K_txt").exists():
        txt_root = txt_root / "DocBank_500K_txt"
    index_dir = source_dir / "indexed_files"
    stats: Dict[str, int] = {}

    for split, index_name in (("train", "500K_train.txt"), ("val", "500K_dev.txt"), ("test", "500K_test.txt")):
        index_path = index_dir / index_name
        if not index_path.exists():
            continue
        records = []
        for line in index_path.read_text().splitlines():
            rel = line.strip()
            if not rel:
                continue
            txt_path = txt_root / rel
            if not txt_path.exists():
                continue
            tokens = []
            blocks_map = defaultdict(list)
            with txt_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    parts = line.strip().split()
                    if len(parts) < 6:
                        continue
                    text = parts[0]
                    bbox = [float(value) for value in parts[1:5]]
                    label = parts[5]
                    tokens.append({"text": text, "bbox": bbox, "block_id": label})
                    blocks_map[label].append(bbox)
            blocks = []
            for idx, (label, bboxes) in enumerate(blocks_map.items()):
                xs = [b[0] for b in bboxes] + [b[2] for b in bboxes]
                ys = [b[1] for b in bboxes] + [b[3] for b in bboxes]
                blocks.append(
                    {
                        "id": f"b{idx}",
                        "type": label.lower(),
                        "bbox": [min(xs), min(ys), max(xs), max(ys)],
                        "children": [],
                    }
                )
            records.append(
                {
                    "id": f"docbank/{split}/{txt_path.stem}",
                    "slug": "docbank",
                    "split": split,
                    "page_index": 0,
                    "image_path": "",
                    "width": 1000,
                    "height": 1000,
                    "tokens": tokens,
                    "blocks": blocks,
                    "classes": [],
                    "metadata": {"source": "docbank", "file": rel},
                }
            )
        stats[split] = _write_jsonl(records, converted_root / "docbank" / f"{split}.jsonl")
    return stats


# Placeholder converters for additional datasets will be appended here.
