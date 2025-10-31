#!/usr/bin/env python3
"""
Render selected pages from a PDF into PNGs (for OCR backends that want images).

Deps:
  - poppler-utils (pdftoppm)  # Fedora: sudo dnf install poppler-utils
  - pdf2image (pip)

Usage:
  python dev/tools/render_pdf_pages.py \
    --pdf "dev/inputs/Basic Skills Review Unit USER.pdf" \
    --out_dir dev/runs/user_e2e/pages \
    --pages 1 7 8 \
    --dpi 220
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Optional
from pdf2image import convert_from_path

def render(pdf: Path, out_dir: Path, pages: Optional[List[int]] = None, dpi: int = 220) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if pages:
        for p in pages:
            imgs = convert_from_path(str(pdf), first_page=p, last_page=p, dpi=dpi, fmt="png")
            img = imgs[0]
            out_path = out_dir / f"page_{p:04d}.png"
            img.save(out_path)
            print(f"[render] wrote {out_path}")
    else:
        imgs = convert_from_path(str(pdf), dpi=dpi, fmt="png")
        for i, img in enumerate(imgs, start=1):
            out_path = out_dir / f"page_{i:04d}.png"
            img.save(out_path)
            print(f"[render] wrote {out_path}")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, help="Path to input PDF")
    ap.add_argument("--out_dir", required=True, help="Directory to write PNG pages")
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--pages", type=int, nargs="*", help="Optional 1-based page numbers")
    args = ap.parse_args()
    render(Path(args.pdf), Path(args.out_dir), args.pages, args.dpi)

if __name__ == "__main__":
    main()
