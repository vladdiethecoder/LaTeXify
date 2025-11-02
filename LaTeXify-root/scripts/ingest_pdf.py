#!/usr/bin/env python3
"""
scripts/ingest_pdf.py

Fallback Stage-1 ingestion (no OCR):
- Reads a PDF
- Extracts per-page text using PyPDF (born-digital text only)
- Optionally exports page images (using pdf-document-layout-analysis helpers)
- Writes a run_dir layout expected by downstream chunker/indexer:
    <run_dir>/
      outputs/fallback/page-0001.md
      outputs/fallback/page-0002.md
      ...
      layout/linked_pages.jsonl
      layout/assets.json (if assets exported)
      meta.json

Notes:
- If you already have OCR outputs, you don't need this.
- This is a deterministic baseline to exercise the full pipeline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from contextlib import suppress
from pathlib import Path
from typing import Dict, List, Sequence

from pypdf import PdfReader

# pdf-document-layout-analysis helpers (optional dependency)
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
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _write_page_md(out_dir: Path, page_index: int, text: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"page-{page_index + 1:04d}.md"
    p = out_dir / name
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    p.write_text(text if text.strip() else f"(empty page {page_index+1})\n", encoding="utf-8")
    return p


def _image_extension(fmt: str | None) -> str:
    fmt = (fmt or "").lower()
    if fmt in {"jpeg", "jpg"}:
        return ".jpg"
    if fmt in {"png"}:
        return ".png"
    if fmt in {"jp2", "jpx"}:
        return ".jp2"
    if fmt:
        return f".{fmt}"
    return ".bin"


def _export_page_images(reader: PdfReader, pdf: Path, assets_dir: Path) -> Sequence[dict]:
    assets_dir.mkdir(parents=True, exist_ok=True)
    manifest: List[dict] = []
    slug = pdf.stem.lower()
    for page_index, page in enumerate(reader.pages, start=1):
        images = getattr(page, "images", [])
        if not images:
            continue
        for image_index, image in enumerate(images, start=1):
            ext = _image_extension(getattr(image, "image_format", None))
            asset_id = f"{slug}-p{page_index:04d}-asset-{image_index:02d}"
            filename = f"{asset_id}{ext}"
            target = assets_dir / filename
            with target.open("wb") as handle:
                handle.write(image.data)
            manifest.append(
                {
                    "id": asset_id,
                    "page": page_index,
                    "type": (getattr(image, "image_format", "image") or "image").lower(),
                    "asset_path": f"assets/{filename}",
                }
            )
    if manifest:
        manifest_path = assets_dir / f"{slug}.manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _export_page_images(pdf: Path, assets_dir: Path) -> List[Dict[str, str]]:
    """Export page-level images using pdf-document-layout-analysis helpers."""

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
            image.save(target)
            exported.append(
                {
                    "page_index": idx,
                    "filename": name,
                    "relative_path": f"{assets_dir.name}/{name}",
                    "asset_path": f"{assets_dir.name}/{name}",
                    "absolute_path": str(target),
                }
            )
    finally:
        if IMAGES_ROOT_PATH is not None:
            with suppress(Exception):
                if IMAGES_ROOT_PATH.exists():
                    PdfImages.remove_images()
    return exported


def _write_manifest(entries: Sequence[Dict[str, str]], manifest_path: Path, assets_dir: Path, pdf: Path) -> None:
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


    if not pdf.exists():
        raise FileNotFoundError(f"PDF not found: {pdf}")
    run_dir.mkdir(parents=True, exist_ok=True)

    reader = PdfReader(str(pdf))
    outputs = run_dir / "outputs" / "fallback"
    pages_written: List[str] = []

    for i, page in enumerate(reader.pages):
        txt = page.extract_text() or ""
        p = _write_page_md(outputs, i, txt)
        pages_written.append(p.name)

    asset_entries: List[Dict[str, str]] = []
    if assets_dir is not None:
        asset_entries = _export_page_images(pdf, assets_dir)
        if manifest_path is not None and asset_entries:
            _write_manifest(asset_entries, manifest_path, assets_dir, pdf)

    # Optional helper: one jsonl entry per page
    (run_dir / "layout").mkdir(parents=True, exist_ok=True)
    with (run_dir / "layout" / "linked_pages.jsonl").open("w", encoding="utf-8") as f:
        for i, name in enumerate(pages_written):
            f.write(json.dumps({"page_index": i, "page_name": name}) + "\n")

    manifest: Sequence[dict] = []
    if assets_dir is not None:
        manifest = _export_page_images(reader, pdf, assets_dir)

    meta = {
        "source_pdf": str(pdf),
        "pdf_sha256": _sha256(pdf),
        "page_count": len(pages_written),
        "backend": "fallback-pypdf",
        "asset_count": len(manifest),
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
    ap = argparse.ArgumentParser(description="Fallback ingestion: PDF → run_dir/outputs/fallback/page-*.md")
    ap.add_argument("--pdf", type=Path, required=True, help="Path to input PDF")
    ap.add_argument("--run_dir", type=Path, required=True, help="Output run directory")
    ap.add_argument(
        "--assets-dir",
        type=Path,
        default=Path("build/assets"),
        help="Directory where extracted figures/tables will be written.",
    )
    args = ap.parse_args()
    ingest_pdf(args.pdf, args.run_dir, assets_dir=args.assets_dir)


if __name__ == "__main__":
    main()
