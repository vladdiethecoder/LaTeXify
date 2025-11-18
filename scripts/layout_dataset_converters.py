#!/usr/bin/env python3
"""Dataset converters that emit LayoutLM-ready JSONL splits."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

TOKEN_SPLIT_RE = re.compile(r"\s+")
DEFAULT_SPLITS = ("train", "val", "test")


def _normalize_bbox(bbox: Iterable[float], width: float, height: float) -> List[int]:
    x, y, w, h = bbox
    x1 = x + w
    y1 = y + h
    return [
        int(max(0, min(1000, round(1000 * x / width)))),
        int(max(0, min(1000, round(1000 * y / height)))),
        int(max(0, min(1000, round(1000 * x1 / width)))),
        int(max(0, min(1000, round(1000 * y1 / height)))),
    ]


def _tokens_from_text(text: str) -> List[str]:
    if not text:
        return []
    return [tok for tok in TOKEN_SPLIT_RE.split(text.strip()) if tok]


def _annotation_label(
    bbox: Iterable[float],
    annotations: Iterable[dict],
    categories: Dict[int, str],
) -> str:
    x, y, w, h = bbox
    center_x = x + w / 2.0
    center_y = y + h / 2.0
    for ann in annotations:
        ax, ay, aw, ah = ann["bbox"]
        if ax <= center_x <= ax + aw and ay <= center_y <= ay + ah:
            return categories.get(ann["category_id"], "O")
    return "O"


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        raise RuntimeError(f"Failed to parse JSON at {path}: {exc}") from exc


def build_doclaynet_splits(source_root: Path, processed_root: Path, *, force: bool = False) -> Dict[str, int]:
    """Convert DocLayNet COCO + JSON extras into LayoutLM JSONL splits."""

    coco_dir = source_root / "DocLayNet_core" / "COCO"
    extra_dir = source_root / "DocLayNet_extra" / "JSON"
    if not coco_dir.exists() or not extra_dir.exists():
        raise FileNotFoundError(
            f"DocLayNet assets missing under {source_root}. Expected 'DocLayNet_core/COCO' and 'DocLayNet_extra/JSON'."
        )
    produced: Dict[str, int] = {}
    for split in DEFAULT_SPLITS:
        coco_path = coco_dir / f"{split}.json"
        if not coco_path.exists():
            raise FileNotFoundError(f"DocLayNet COCO file missing: {coco_path}")
        output_path = processed_root / "splits" / split / "data.jsonl"
        if output_path.exists() and not force:
            produced[split] = sum(1 for _ in output_path.open("r", encoding="utf-8"))
            continue
        print(f"[doclaynet] Generating {split} split from {coco_path}")
        data = _load_json(coco_path)
        images = {img["id"]: img for img in data.get("images", [])}
        categories = {cat["id"]: cat["name"] for cat in data.get("categories", [])}
        annotations = defaultdict(list)
        for ann in data.get("annotations", []):
            annotations[ann["image_id"]].append(ann)
        count = 0
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for image_id, image in images.items():
                file_name = image["file_name"]
                page_stem = Path(file_name).stem
                extra_path = extra_dir / f"{page_stem}.json"
                if not extra_path.exists():
                    continue
                extra_payload = _load_json(extra_path)
                cells = extra_payload.get("cells", [])
                width = float(image.get("width") or extra_payload.get("metadata", {}).get("coco_width") or 1025)
                height = float(image.get("height") or extra_payload.get("metadata", {}).get("coco_height") or 1025)
                tokens: List[str] = []
                bboxes: List[List[int]] = []
                ner_tags: List[str] = []
                for cell in cells:
                    text = cell.get("text", "")
                    cell_tokens = _tokens_from_text(text)
                    if not cell_tokens:
                        continue
                    bbox = cell.get("bbox")
                    if not bbox:
                        continue
                    label = _annotation_label(bbox, annotations[image_id], categories)
                    normalized = _normalize_bbox(bbox, width, height)
                    for token in cell_tokens:
                        tokens.append(token)
                        bboxes.append(normalized)
                        ner_tags.append(label)
                if not tokens:
                    continue
                example = {
                    "id": page_stem,
                    "tokens": tokens,
                    "bboxes": bboxes,
                    "ner_tags": ner_tags,
                }
                handle.write(json.dumps(example) + "\n")
                count += 1
        produced[split] = count
        print(f"[doclaynet] Wrote {count} samples to {output_path}")
    return produced


def _iter_docbank_files(base_dir: Path, index_file: Path | None = None) -> Iterable[Path]:
    if index_file and index_file.exists():
        with index_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                name = line.strip()
                if not name:
                    continue
                candidate = base_dir / name
                if candidate.is_file():
                    yield candidate
        return
    if not base_dir.exists():
        return []
    return sorted(p for p in base_dir.rglob("*.txt") if p.is_file())


def _parse_docbank_txt(path: Path) -> tuple[list[str], list[list[int]], list[str]]:
    tokens: List[str] = []
    bboxes: List[List[int]] = []
    labels: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            parts = raw.split()
            if len(parts) < 6:
                continue
            token = parts[0]
            try:
                coords = list(map(float, parts[1:5]))
            except ValueError:
                continue
            label = parts[5]
            tokens.append(token)
            bboxes.append(_normalize_bbox(coords, 1000.0, 1000.0))
            labels.append(label)
    return tokens, bboxes, labels


def build_docbank_splits(source_root: Path, processed_root: Path, *, force: bool = False) -> Dict[str, int]:
    txt_root = source_root / "DocBank_500K_txt"
    if not txt_root.exists():
        raise FileNotFoundError(f"DocBank text directory missing at {txt_root}")
    if (txt_root / "DocBank_500K_txt").exists():
        txt_root = txt_root / "DocBank_500K_txt"
    index_dir = source_root / "indexed_files"
    split_indices = {
        "train": index_dir / "500K_train.txt",
        "val": index_dir / "500K_dev.txt",
        "test": index_dir / "500K_test.txt",
    }
    produced: Dict[str, int] = {}
    for split in DEFAULT_SPLITS:
        output_path = processed_root / "splits" / split / "data.jsonl"
        if output_path.exists() and not force:
            produced[split] = sum(1 for _ in output_path.open("r", encoding="utf-8"))
            continue
        index_file = split_indices.get(split)
        print(f"[docbank] Generating {split} split")
        files = _iter_docbank_files(txt_root, index_file)
        count = 0
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for txt_path in files:
                tokens, bboxes, labels = _parse_docbank_txt(txt_path)
                if not tokens:
                    continue
                example = {
                    "id": txt_path.stem,
                    "tokens": tokens,
                    "bboxes": bboxes,
                    "ner_tags": labels,
                }
                handle.write(json.dumps(example) + "\n")
                count += 1
        produced[split] = count
        print(f"[docbank] Wrote {count} samples to {output_path}")
    return produced


CONVERTERS = {
    "doclaynet": build_doclaynet_splits,
    "docbank": build_docbank_splits,
}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Path to the raw dataset directory (e.g., training_data/raw/doclaynet/source).")
    parser.add_argument("--processed", required=True, help="Path to the processed dataset root (e.g., training_data/processed/doclaynet).")
    parser.add_argument("--slug", required=True, choices=sorted(CONVERTERS.keys()))
    parser.add_argument("--force", action="store_true", help="Rebuild even if data.jsonl already exists.")
    args = parser.parse_args()

    converter = CONVERTERS[args.slug]
    converter(Path(args.source), Path(args.processed), force=args.force)


if __name__ == "__main__":
    main()
