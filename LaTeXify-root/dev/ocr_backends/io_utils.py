# -*- coding: utf-8 -*-
"""
IO helpers for OCR backends.

- load_page_image(): accepts a path to an image OR to a PDF (with 1-based page).
- If the input path doesn't exist, we try to find likely candidates and emit
  actionable error messages (with suggestions).

Requires:
  - Pillow (PIL)
  - For PDF rasterization: pdf2image + poppler (install poppler-utils on Linux).
"""
from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import List, Optional

from PIL import Image


def _is_pdf(path: Path) -> bool:
    return path.suffix.lower() == ".pdf"


def _list_candidates(page: int) -> List[Path]:
    """
    Heuristics to surface likely page images in your dev/runs structure.
    """
    pats = [
        f"dev/runs/**/pages/page_{page:04d}.png",
        f"dev/runs/**/pages/page_{page:04d}.jpg",
        f"dev/runs/**/pages/*{page:04d}*.png",
        f"dev/runs/**/pages/*{page:04d}*.jpg",
    ]
    hits: List[Path] = []
    for pat in pats:
        hits.extend(Path(p).resolve() for p in glob.glob(pat, recursive=True))
    return hits


def _list_known_pdfs() -> List[Path]:
    pats = [
        "dev/inputs/*.pdf",
        "dev/inputs/**/*.pdf",
        "dev/**/inputs/*.pdf",
        "dev/**/inputs/**/*.pdf",
    ]
    hits: List[Path] = []
    for pat in pats:
        hits.extend(Path(p).resolve() for p in glob.glob(pat, recursive=True))
    # Also common place when running e2e:
    for run_dir in Path("dev/runs").glob("*_e2e"):
        hits.extend(Path(p).resolve() for p in (run_dir / "inputs").glob("*.pdf"))
    return hits


def _helpful_not_found(kind: str, missing: Path, page: int) -> FileNotFoundError:
    tips = []

    if kind == "pdf":
        tips.append("• The path does not exist. Verify the exact filename under dev/inputs/.")
        pdfs = _list_known_pdfs()[:10]
        if pdfs:
            tips.append("• Examples I found:")
            tips.extend([f"  - {p}" for p in pdfs])
        tips.append("• If you only have images, point to a page PNG/JPG instead.")
        msg = (
            f"PDF not found: {missing}\n" + "\n".join(tips) +
            "\n(If you want to open a PDF directly, ensure pdf2image+poppler are installed.)"
        )
        return FileNotFoundError(msg)

    # kind == "image"
    tips.append("• The image path does not exist.")
    cand = _list_candidates(page)
    if cand:
        tips.append("• Did you mean one of these?")
        tips.extend([f"  - {p}" for p in cand[:10]])
    else:
        tips.append("• I couldn't find page images under dev/runs/**/pages/. "
                    "If you have a PDF, pass that instead and set --page.")
    msg = f"Image not found: {missing}\n" + "\n".join(tips)
    return FileNotFoundError(msg)


def _load_image(path: Path) -> Image.Image:
    if not path.exists():
        raise _helpful_not_found("image", path, page=1)
    return Image.open(str(path)).convert("RGB")


def _pdf_page_to_image(pdf_path: Path, page: int, dpi: int = 200) -> Image.Image:
    try:
        from pdf2image import convert_from_path  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "pdf2image is required to open PDF pages.\n"
            "Install with: pip install pdf2image\n"
            "Linux (Fedora): sudo dnf install poppler-utils\n"
            "Linux (Ubuntu/Debian): sudo apt-get install poppler-utils\n"
        ) from e

    if not pdf_path.exists():
        raise _helpful_not_found("pdf", pdf_path, page=page)

    images = convert_from_path(str(pdf_path), dpi=dpi, first_page=page, last_page=page)
    if not images:
        raise ValueError(f"Failed to render page {page} from {pdf_path}")
    return images[0].convert("RGB")


def load_page_image(path_or_pdf: str, page: int = 1, dpi: int = 200) -> Image.Image:
    """
    Load an RGB PIL.Image from an image path or from a given page of a PDF.
    If the path doesn't exist, emit a useful error with suggestions.
    """
    p = Path(path_or_pdf)
    if _is_pdf(p):
        return _pdf_page_to_image(p, page=page, dpi=dpi)
    if not p.exists():
        # Try to guess likely page images and suggest them.
        raise _helpful_not_found("image", p, page=page)
    return _load_image(p)
