#!/usr/bin/env python3
"""Replace placeholder layout splits with real tokens/bboxes/labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DATA = REPO_ROOT / "training_data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slug", required=True, help="Dataset slug (matches training_data/processed/<slug>).")
    parser.add_argument("--train-file", help="Path to JSON/JSONL file for the train split.")
    parser.add_argument("--val-file", help="Path to JSON/JSONL file for the val split.")
    parser.add_argument("--test-file", help="Path to JSON/JSONL file for the test split.")
    parser.add_argument(
        "--replace-placeholders",
        action="store_true",
        help="Delete existing placeholder files before writing splits.",
    )
    return parser.parse_args()


def validate_records(path: Path) -> None:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} line {line_no} is not valid JSON: {exc}") from exc
            for key in ("tokens", "bboxes", "ner_tags"):
                if key not in record:
                    raise ValueError(f"{path} line {line_no} missing '{key}' field.")
            if len(record["tokens"]) != len(record["bboxes"]) or len(record["tokens"]) != len(record["ner_tags"]):
                raise ValueError(f"{path} line {line_no} has mismatched tokens/bboxes/ner_tags lengths.")


def write_split(target: Path, source: Path | str, replace: bool, slug: str, split: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not replace:
        raise FileExistsError(f"{target} already exists. Pass --replace-placeholders to overwrite.")
    if isinstance(source, Path):
        validate_records(source)
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        sample = {
            "id": f"{slug}-{split}-placeholder",
            "tokens": ["Placeholder"],
            "bboxes": [[0, 0, 10, 10]],
            "ner_tags": ["O"],
        }
        target.write_text(json.dumps(sample) + "\n", encoding="utf-8")
    print(f"[register] wrote split '{target.relative_to(REPO_ROOT)}'")


def main() -> None:
    args = parse_args()
    slug_dir = TRAINING_DATA / "processed" / args.slug
    if not slug_dir.exists():
        print(f"[register] creating missing slug directory at {slug_dir}")
        slug_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        (slug_dir / "splits" / split).mkdir(parents=True, exist_ok=True)
    mapping = {
        "train": args.train_file,
        "val": args.val_file,
        "test": args.test_file,
    }
    if not any(mapping.values()):
        raise ValueError("Provide at least one of --train-file/--val-file/--test-file.")
    for split, source_path in mapping.items():
        if not source_path:
            continue
        if source_path == "placeholder":
            target = slug_dir / "splits" / split / "data.jsonl"
            write_split(target, source_path, args.replace_placeholders, args.slug, split)
            continue
        source = Path(source_path).expanduser()
        if not source.exists():
            print(f"[register] {source} not found; writing placeholder split instead.")
            target = slug_dir / "splits" / split / "data.jsonl"
            write_split(target, "placeholder", args.replace_placeholders, args.slug, split)
        else:
            target = slug_dir / "splits" / split / "data.jsonl"
            write_split(target, source, args.replace_placeholders, args.slug, split)
    print("[register] done.")


if __name__ == "__main__":
    main()
