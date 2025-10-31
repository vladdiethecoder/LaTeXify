#!/usr/bin/env python3
"""
scripts/ingest_pdf.py

Fallback Stage-1 ingestion (no OCR):
- Reads a PDF
- Extracts per-page text using PyPDF (born-digital text only)
- Writes a run_dir layout expected by downstream chunker/indexer:
    <run_dir>/
      outputs/fallback/page-0001.md
      outputs/fallback/page-0002.md
      ...
      layout/linked_pages.jsonl
      meta.json

Notes:
- If you already have OCR outputs, you don't need this.
- This is a deterministic baseline to exercise the full pipeline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import List

from pypdf import PdfReader


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


def ingest_pdf(pdf: Path, run_dir: Path) -> None:
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

    # Optional helper: one jsonl entry per page
    (run_dir / "layout").mkdir(parents=True, exist_ok=True)
    with (run_dir / "layout" / "linked_pages.jsonl").open("w", encoding="utf-8") as f:
        for i, name in enumerate(pages_written):
            f.write(json.dumps({"page_index": i, "page_name": name}) + "\n")

    meta = {
        "source_pdf": str(pdf),
        "pdf_sha256": _sha256(pdf),
        "page_count": len(pages_written),
        "backend": "fallback-pypdf",
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[ingest] Wrote {len(pages_written)} pages → {outputs}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Fallback ingestion: PDF → run_dir/outputs/fallback/page-*.md")
    ap.add_argument("--pdf", type=Path, required=True, help="Path to input PDF")
    ap.add_argument("--run_dir", type=Path, required=True, help="Output run directory")
    args = ap.parse_args()
    ingest_pdf(args.pdf, args.run_dir)


if __name__ == "__main__":
    main()
