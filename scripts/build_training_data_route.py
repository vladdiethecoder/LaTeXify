#!/usr/bin/env python3
"""Generate a structured training-data route from training_database-catalog.json.

The script reads the catalog (JSON lines), derives a storage layout for every
dataset, creates placeholder directories under training_data/raw|processed, and
writes manifest files that describe how to pull/scrape each source.
"""

from __future__ import annotations

import argparse
import json
import re
from urllib.parse import urlparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = REPO_ROOT / "training_database-catalog.json"
TRAINING_DATA_DIR = REPO_ROOT / "training_data"

MODALITY_DIR_MAP = {
    "pdf": "pdf",
    "page-image": "images",
    "json": "annotations",
    "latex-src": "latex",
    "text": "text",
}

TRACK_DESCRIPTIONS = {
    "T1": "Document layout + form understanding",
    "T2": "Text detection in natural/scene imagery",
    "T3": "OCR and handwriting recognition",
    "T4": "Table detection + structure recovery",
    "T5": "Chart/diagram reasoning",
    "T6": "Document QA and key-value extraction",
    "T8": "Text-to-LaTeX / PDF infill",
}


@dataclass
class IngestPlan:
    strategy: str
    command: str | None
    notes: List[str]
    needs_auth: bool


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "dataset"


def relpath(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT).as_posix())


def safe_relpath(path: Path) -> str:
    try:
        return relpath(path)
    except ValueError:
        return str(path)


def ensure_placeholder_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    placeholder = path / ".gitkeep"
    if not placeholder.exists():
        placeholder.touch()


def _hf_repo_from_url(url: str) -> str | None:
    parsed = urlparse(url)
    if "huggingface.co" not in parsed.netloc:
        return None
    path = parsed.path.strip("/")
    if not path:
        return None
    parts = path.split("/")
    if parts[0] in {"datasets", "models", "spaces"} and len(parts) >= 3:
        return "/".join(parts[1:3])
    return "/".join(parts[:2])


def build_ingest_plan(url: str, dest: Path) -> IngestPlan:
    dest_str = relpath(dest)
    if "github.com" in url:
        command = f"git clone {url} \"{dest_str}\""
        return IngestPlan(strategy="git", command=command, notes=[], needs_auth=False)
    if "huggingface.co" in url:
        repo = _hf_repo_from_url(url)
        if repo:
            command = (
                "hf download "
                f"{repo} --repo-type dataset --local-dir \"{dest_str}\""
            )
            return IngestPlan(
                strategy="huggingface-datasets",
                command=command,
                notes=[],
                needs_auth=True,
            )
        command = (
            "Manual Hugging Face retrieval required; "
            "could not parse dataset repo id from URL."
        )
        return IngestPlan(strategy="manual", command=None, notes=[command], needs_auth=True)
    notes = [
        "Manual download required",
        f"Visit {url} and follow the provider instructions.",
    ]
    return IngestPlan(strategy="manual", command=None, notes=notes, needs_auth=True)


def iter_catalog_entries(catalog_path: Path) -> Iterable[Dict]:
    text = catalog_path.read_text().splitlines()
    for line in text:
        stripped = line.strip()
        if not stripped or stripped.startswith("Total output"):
            continue
        yield json.loads(stripped)


def unique_tracks(entries: Sequence[Dict]) -> Dict[str, Dict]:
    mapping = {}
    for key, description in TRACK_DESCRIPTIONS.items():
        mapping[key] = {"description": description, "datasets": []}
    for entry in entries:
        for track in entry.get("tracks", []):
            mapping.setdefault(track, {"description": "", "datasets": []})
            mapping[track]["datasets"].append(entry["name"])
    for value in mapping.values():
        value["datasets"].sort()
    return mapping


def build_manifest(entry: Dict) -> Dict:
    slug = slugify(entry["name"])
    raw_root = TRAINING_DATA_DIR / "raw" / slug
    processed_root = TRAINING_DATA_DIR / "processed" / slug
    ensure_placeholder_dir(raw_root)
    ensure_placeholder_dir(processed_root)

    ingest = build_ingest_plan(entry.get("url", ""), raw_root / "source")

    modality_plan = []
    for modality in entry.get("modality", []):
        subdir = MODALITY_DIR_MAP.get(modality, modality)
        modality_plan.append(
            {
                "modality": modality,
                "path": relpath(raw_root / subdir),
            }
        )

    manifest = {
        "name": entry["name"],
        "slug": slug,
        "tracks": entry.get("tracks", []),
        "url": entry.get("url"),
        "license": entry.get("license"),
        "storage": {
            "raw_root": relpath(raw_root),
            "processed_root": relpath(processed_root),
            "modalities": modality_plan,
        },
        "ingest": {
            "strategy": ingest.strategy,
            "command": ingest.command,
            "needs_auth": ingest.needs_auth,
            "notes": ingest.notes,
        },
        "preprocess": entry.get("preprocess", []),
        "strengths": entry.get("strengths", []),
        "limitations": entry.get("limitations", []),
        "modality": entry.get("modality", []),
        "domain": entry.get("domain", []),
        "is_synthetic": entry.get("is_synthetic"),
        "languages": entry.get("languages", []),
        "size": entry.get("size"),
        "relevance": entry.get("relevance"),
        "justification": entry.get("justification"),
    }
    return manifest


def write_manifest(manifest: Dict) -> Path:
    manifests_dir = TRAINING_DATA_DIR / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    path = manifests_dir / f"{manifest['slug']}.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def build_route(catalog_path: Path) -> Dict:
    entries = list(iter_catalog_entries(catalog_path))
    manifests = []
    for entry in entries:
        manifest = build_manifest(entry)
        write_manifest(manifest)
        manifests.append(manifest)

    route = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "catalog": safe_relpath(catalog_path),
        "dataset_count": len(manifests),
        "datasets": [
            {
                "name": item["name"],
                "slug": item["slug"],
                "tracks": item["tracks"],
                "raw_root": item["storage"]["raw_root"],
                "processed_root": item["storage"]["processed_root"],
                "ingest_strategy": item["ingest"]["strategy"],
            }
            for item in manifests
        ],
        "tracks": unique_tracks(entries),
    }
    return route


def write_route(route: Dict) -> Path:
    TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = TRAINING_DATA_DIR / "route.json"
    path.write_text(json.dumps(route, indent=2) + "\n")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create training data route/manifests.")
    parser.add_argument(
        "--catalog",
        default=str(DEFAULT_CATALOG),
        help="Path to training_database-catalog.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    catalog_path = Path(args.catalog).resolve()
    if not catalog_path.exists():
        raise SystemExit(f"Catalog file not found: {catalog_path}")
    route = build_route(catalog_path)
    route_path = write_route(route)
    print(f"Wrote training route: {route_path}")


if __name__ == "__main__":
    main()
