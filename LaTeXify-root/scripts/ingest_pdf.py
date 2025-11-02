#!/usr/bin/env python3
"""
Fallback Stage‑1 ingestion (no OCR) for the LaTeXify pipeline.

This script reads a PDF, extracts the born‑digital text from each page using
``pypdf``, optionally exports rasterised page images via
``pdf-document-layout-analysis`` helpers and writes a run directory in the
expected format for downstream stages.  The run directory will contain:

    <run_dir>/
      outputs/fallback/page-0001.md
      outputs/fallback/page-0002.md
      ...
      layout/linked_pages.jsonl
      layout/assets.json (if assets exported)
      meta.json

If you already have OCR outputs, you do not need this ingestion step.  This
stage provides a deterministic baseline to exercise the full pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import suppress
from pathlib import Path
from typing import Dict, List, Sequence

import sys
from pypdf import PdfReader

# Attempt to import pdf-document-layout-analysis helpers.  These are optional
# dependencies; if unavailable, asset export will be skipped.
REPO_ROOT = Path(__file__).resolve().parents[1]
PDLA_SRC = REPO_ROOT / "pdf-document-layout-analysis" / "src"
if str(PDLA_SRC) not in sys.path:
    sys.path.append(str(PDLA_SRC))
try:  # pragma: no cover - optional import
    from domain.PdfImages import PdfImages  # type: ignore
    from configuration import IMAGES_ROOT_PATH  # type: ignore
except Exception:  # pragma: no cover - graceful fallback if dependency missing
    PdfImages = None  # type: ignore
    IMAGES_ROOT_PATH = None  # type: ignore

DEFAULT_BUILD = REPO_ROOT / "build"
DEFAULT_ASSET_DIR = DEFAULT_BUILD / "assets"


def _sha256(path: Path) -> str:
    """Compute the SHA‑256 digest of a file and return a tag string."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _write_page_md(out_dir: Path, page_index: int, text: str) -> Path:
    """Write extracted page text to a markdown file in the fallback output dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"page-{page_index + 1:04d}.md"
    p = out_dir / name
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    p.write_text(text if text.strip() else f"(empty page {page_index+1})\n", encoding="utf-8")
    return p


def _export_page_images(pdf: Path, assets_dir: Path) -> List[Dict[str, str]]:
    """Export page‑level images using pdf‑document‑layout‑analysis helpers.

    When the optional dependency ``pdf-document-layout-analysis`` is installed,
    this function will rasterise each page of the PDF into a PNG file and
    return a manifest describing the exported assets.  If the dependency is
    missing, an empty list is returned and a message is printed to stderr.
    """
    if PdfImages is None:
        print("[ingest] pdf-document-layout-analysis unavailable; skipping asset export")
        return []
    assets_dir.mkdir(parents=True, exist_ok=True)
    exported: List[Dict[str, str]] = []
    pdf_images = PdfImages.from_pdf_path(str(pdf))
    try:
        for idx, image in enumerate(pdf_images.pdf_images):
            name = f"{pdf.stem}-page-{idx + 1:04d}.png"
            target = assets_dir / name
            # Each image exposes a PIL-like API with a save() method
            image.save(target)
            exported.append(
                {
                    "page_index": idx,
                    "filename": name,
                    "asset_id": f"{pdf.stem}-page-{idx + 1:04d}",
                    "type": "page_image",
                    "source": "page_render",
                    "relative_path": f"{assets_dir.name}/{name}",
                    "asset_path": f"{assets_dir.name}/{name}",
                    "absolute_path": str(target),
                }
            )
    finally:
        # Remove temporary files produced by PdfImages to avoid disk bloat
        if IMAGES_ROOT_PATH is not None:
            with suppress(Exception):
                if IMAGES_ROOT_PATH.exists():
                    PdfImages.remove_images()
    return exported


def _write_manifest(entries: Sequence[Dict[str, str]], manifest_path: Path, assets_dir: Path, pdf: Path) -> None:
    """Write an asset manifest JSON file describing exported images."""
    payload = {
        "source_pdf": str(pdf),
        "asset_dir": str(assets_dir),
        "asset_dir_name": assets_dir.name,
        "entries": list(entries),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def ingest_pdf(
    pdf: Path,
    run_dir: Path,
    *,
    assets_dir: Path | None = None,
    manifest_path: Path | None = None,
) -> None:
    """Ingest a PDF into a run directory and optionally export page images."""
    if not pdf.exists():
        raise FileNotFoundError(f"PDF not found: {pdf}")
    run_dir.mkdir(parents=True, exist_ok=True)
    reader = PdfReader(str(pdf))
    outputs = run_dir / "outputs" / "fallback"
    pages_written: List[str] = []
    # Extract text from each page and write to markdown
    for i, page in enumerate(reader.pages):
        txt = page.extract_text() or ""
        p = _write_page_md(outputs, i, txt)
        pages_written.append(p.name)
    # Export assets once if requested
    asset_entries: List[Dict[str, str]] = []
    if assets_dir is not None:
        asset_entries = _export_page_images(pdf, assets_dir)
        if manifest_path is not None and asset_entries:
            _write_manifest(asset_entries, manifest_path, assets_dir, pdf)
    # Write linked_pages.jsonl mapping page indices to filenames
    (run_dir / "layout").mkdir(parents=True, exist_ok=True)
    with (run_dir / "layout" / "linked_pages.jsonl").open("w", encoding="utf-8") as f:
        for i, name in enumerate(pages_written):
            f.write(json.dumps({"page_index": i, "page_name": name}) + "\n")
    # Build meta.json summarising the run
    meta: Dict[str, object] = {
        "source_pdf": str(pdf),
        "pdf_sha256": _sha256(pdf),
        "page_count": len(pages_written),
        "backend": "fallback-pypdf",
        "asset_count": len(asset_entries),
    }
    if asset_entries:
        meta["assets"] = {
            "dir": str(assets_dir),
            "relative_dir": assets_dir.name,
            "entries": asset_entries,
        }
        if manifest_path is not None:
            meta["asset_manifest"] = str(manifest_path)
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    summary = {
        "pages": len(pages_written),
        "outputs": str(outputs),
        "assets_exported": len(asset_entries),
        "assets_dir": str(assets_dir) if asset_entries else None,
    }
    print(f"[ingest] {json.dumps(summary)}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fallback ingestion: PDF → run_dir/outputs/fallback/page-*.md"
    )
    ap.add_argument("--pdf", type=Path, required=True, help="Path to input PDF")
    ap.add_argument("--run_dir", type=Path, required=True, help="Output run directory")
    ap.add_argument(
        "--assets-dir",
        type=Path,
        default=DEFAULT_ASSET_DIR,
        help="Directory to export detected images (default: build/assets)",
    )
    ap.add_argument(
        "--asset-manifest",
        type=Path,
        default=None,
        help="Optional path to write a JSON manifest describing exported assets",
    )
    args = ap.parse_args()
    manifest_path = args.asset_manifest or (args.run_dir / "layout" / "assets.json")
    ingest_pdf(args.pdf, args.run_dir, assets_dir=args.assets_dir, manifest_path=manifest_path)


if __name__ == "__main__":
    main()