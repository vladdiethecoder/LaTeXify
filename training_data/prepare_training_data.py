"""CLI entrypoint for downloading + converting LayoutJSONL datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from .downloaders import ensure_downloaded
from .layout_dataset_converters import CONVERTERS

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = REPO_ROOT / "training_data"
MANIFEST_DIR = TRAINING_DIR / "manifests"
CONVERTED_ROOT = TRAINING_DIR / "converted"


def load_manifests() -> List[Tuple[str, dict]]:
    manifests = []
    for manifest_path in sorted(MANIFEST_DIR.glob("*.json")):
        data = json.loads(manifest_path.read_text())
        manifests.append((manifest_path.stem, data))
    return manifests


def run(slug: str, manifest: dict, *, force: bool = False) -> Dict[str, int]:
    if manifest.get("blocked"):
        reason = manifest.get("reason", "blocked")
        print(f"[skip] {slug}: {reason}")
        return {}

    raw_root = REPO_ROOT / manifest["storage"]["raw_root"]
    raw_root.mkdir(parents=True, exist_ok=True)
    ensure_downloaded(slug, manifest, raw_root)
    converter = CONVERTERS.get(slug)
    if converter is None:
        print(f"[warn] {slug}: no converter registered; skipping LayoutJSONL emission.")
        return {}
    converted_dir = CONVERTED_ROOT
    stats = converter(raw_root, converted_dir)
    return stats


def summarize(all_stats: Dict[str, Dict[str, int]]) -> None:
    header = f"{'Slug':20} | {'Split':8} | {'Records':8}"
    print(header)
    print("-" * len(header))
    for slug, stats in all_stats.items():
        if not stats:
            print(f"{slug:20} | {'-':8} | {'0':8}")
            continue
        for split, count in stats.items():
            print(f"{slug:20} | {split:8} | {count:8d}")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slugs", help="Comma-separated slugs to limit conversion.")
    parser.add_argument("--force", action="store_true", help="Force rebuilding outputs.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    selected = set(args.slugs.split(",")) if args.slugs else None
    summaries: Dict[str, Dict[str, int]] = {}
    for _, manifest in load_manifests():
        slug = manifest["slug"]
        if selected and slug not in selected:
            continue
        stats = run(slug, manifest, force=args.force)
        summaries[slug] = stats
    summarize(summaries)


if __name__ == "__main__":
    main()
