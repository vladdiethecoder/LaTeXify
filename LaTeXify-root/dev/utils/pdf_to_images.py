# dev/utils/pdf_to_images.py
from __future__ import annotations
import pathlib
import fitz  # PyMuPDF
from typing import List

def pdf_to_images(pdf_path: str, out_dir: str, dpi: int = 400, prefix: str = "page") -> List[str]:
    """
    Convert all pages of a PDF to PNGs at the requested DPI.
    Returns the list of image paths in reading order.
    """
    pdf = pathlib.Path(pdf_path)
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(str(pdf))
    paths: List[str] = []
    for i in range(doc.page_count):
        pix = doc[i].get_pixmap(dpi=dpi)  # 300+ dpi improves OCR fidelity
        img_path = out / f"{prefix}-{i+1:04d}.png"
        pix.save(img_path)
        paths.append(str(img_path))
    return paths
