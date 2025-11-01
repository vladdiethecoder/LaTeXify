from __future__ import annotations
import subprocess
from pathlib import Path
from typing import List

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _pdftoppm(pdf: Path, out_dir: Path, dpi: int) -> List[Path]:
    _ensure_dir(out_dir)
    # pdftoppm writes page-000001.png, etc.
    prefix = out_dir / "page"
    cmd = ["pdftoppm", "-png", "-r", str(dpi), str(pdf), str(prefix)]
    subprocess.run(cmd, check=True)
    paths = sorted(out_dir.glob("page-*.png"))
    if not paths:
        # some pdftoppm versions use page-000001.png or page-1.png
        paths = sorted(out_dir.glob("page*.png"))
    return [p.resolve() for p in paths]

def pdf_to_images(pdf_path: str | Path, out_dir: str | Path, dpi: int = 300) -> List[Path]:
    """
    Convert a PDF to per-page PNG images and return their absolute paths.

    Prefers pdf2image (Pillow pipeline). Falls back to system 'pdftoppm'
    for robustness.
    """
    pdf = Path(pdf_path).resolve()
    out = Path(out_dir).resolve()
    _ensure_dir(out)
    # Try pdf2image first
    try:
        from pdf2image import convert_from_path  # type: ignore
        images = convert_from_path(str(pdf), dpi=dpi)
        paths: List[Path] = []
        for i, img in enumerate(images, start=1):
            p = out / f"page_{i:04d}.png"
            img.save(str(p), "PNG")
            paths.append(p.resolve())
        if paths:
            return paths
    except Exception:
        pass  # fall back to pdftoppm

    # Fallback to pdftoppm
    return _pdftoppm(pdf, out, dpi)
