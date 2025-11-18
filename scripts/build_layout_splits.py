#!/usr/bin/env python3
"""CLI to rebuild LayoutLM-ready JSONL splits from raw layout datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training_data.layout_dataset_converters import CONVERTERS

RAW_ROOT = REPO_ROOT / "training_data" / "raw"
CONVERTED_ROOT = REPO_ROOT / "training_data" / "converted"


def resolve_slugs(arg: str | None, use_all: bool) -> List[str]:
    if use_all:
        return sorted(CONVERTERS.keys())
    if not arg:
        raise SystemExit("Specify --slugs comma-separated or pass --all.")
    return [slug.strip() for slug in arg.split(",") if slug.strip()]


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slugs", help="Comma-separated slug list (default: none).")
    parser.add_argument("--all", action="store_true", help="Build splits for every supported slug.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    slugs = resolve_slugs(args.slugs, args.all)
    for slug in slugs:
        converter = CONVERTERS.get(slug)
        if converter is None:
            raise SystemExit(f"Slug '{slug}' is not supported. Available: {', '.join(sorted(CONVERTERS))}")
        raw_root = RAW_ROOT / slug
        converted_root = CONVERTED_ROOT
        if not raw_root.exists():
            raise SystemExit(f"Raw dataset for '{slug}' not found at {raw_root}. Run prepare-training-data first.")
        print(f"[build] {slug}: raw_root={raw_root}")
        converter(raw_root, converted_root)


if __name__ == "__main__":
    main()
