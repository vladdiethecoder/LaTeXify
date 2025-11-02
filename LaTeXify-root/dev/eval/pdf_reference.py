"""
Utilities for extracting textual and structural information from PDFs and
computing a simple perceptual distance between two documents.

This module exposes two primary functions:

* ``extract_pdf_reference``: Given a path to a PDF, it returns a list of
  ``PageRef`` objects, each containing the plain text for that page and a
  list of text blocks with bounding boxes.  This is useful when performing
  regression tests on the textual layout of generated documents.

* ``compute_dssim_for_pdfs``: Given two PDFs, it renders each page into a
  greyscale numpy array using PyMuPDF and computes the DSSIM (1 – SSIM)/2
  between corresponding pages.  The Structural Similarity Index (SSIM)
  provides a simple measure of perceptual similarity between images.  A
  DSSIM of 0.0 indicates identical images while values closer to 1.0
  indicate greater dissimilarity.

The reference implementation originally included a separate code path
using ``pdf2image`` and Pillow for rendering.  That implementation
proved brittle and introduced additional dependencies.  The current
implementation therefore relies solely on PyMuPDF for rendering PDF pages
to images.  If you need the ``pdf2image`` based implementation, refer to
earlier commits in the repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import fitz  # type: ignore  # PyMuPDF
import numpy as np


@dataclass
class PDFImageComparison:
    """Container for DSSIM comparison results between two PDFs."""

    reference: str
    candidate: str
    page_dssim: List[float]
    mean_dssim: float


@dataclass
class PageRef:
    """Represents the textual content and layout blocks for a single PDF page."""

    text: str
    blocks: List[Dict[str, Any]]  # raw blocks with bbox and text


def extract_pdf_reference(pdf_path: str) -> List[PageRef]:
    """Extract text and block-level information from every page in a PDF.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        A list of ``PageRef`` objects, one per page in the document.
    """
    doc = fitz.open(pdf_path)
    out: List[PageRef] = []
    for p in doc:
        # text (layout-preserving to a degree)
        text = p.get_text("text")
        # blocks: [(x0, y0, x1, y1, "text", block_no, block_type, ...)]
        blocks_raw = p.get_text("blocks")
        blocks = []
        for b in blocks_raw:
            blocks.append({
                "bbox": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                "text": (b[4] or "").strip(),
            })
        out.append(PageRef(text=text, blocks=blocks))
    return out


def _pixmap_to_gray(pix: fitz.Pixmap) -> np.ndarray:
    """Convert a PyMuPDF Pixmap to a normalised greyscale numpy array."""
    # If the pixmap has an alpha channel, drop it to obtain consistent shape.
    if pix.alpha:
        pix = fitz.Pixmap(fitz.csRGB, pix)
    data = np.frombuffer(pix.samples, dtype=np.uint8)
    arr = data.reshape(pix.height, pix.width, pix.n)
    if pix.n == 1:
        gray = arr[:, :, 0]
    else:
        rgb = arr[:, :, :3]
        gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    # Normalise to [0, 1]
    return (gray.astype(np.float32) / 255.0).copy()


def render_pdf_to_images(pdf_path: str, dpi: int = 144) -> List[np.ndarray]:
    """Render all pages of a PDF into greyscale images.

    Args:
        pdf_path: Path to the PDF.
        dpi: Dots per inch used for rendering (higher values yield larger images).

    Returns:
        A list of 2D numpy arrays representing greyscale page images.
    """
    doc = fitz.open(pdf_path)
    images: List[np.ndarray] = []
    for page in doc:
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        images.append(_pixmap_to_gray(pix))
    return images


def structural_similarity(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """Compute the Structural Similarity Index (SSIM) between two greyscale images.

    The SSIM is computed on the overlapping region of the two images.  If the
    shapes differ, the images are cropped to the minimum common height and
    width.  The result lies in the range [0, 1], where 1 indicates identical
    images.

    Args:
        img_a: First image array.
        img_b: Second image array.

    Returns:
        The SSIM value.
    """
    if img_a.size == 0 or img_b.size == 0:
        return 0.0
    if img_a.shape != img_b.shape:
        min_h = min(img_a.shape[0], img_b.shape[0])
        min_w = min(img_a.shape[1], img_b.shape[1])
        img_a = img_a[:min_h, :min_w]
        img_b = img_b[:min_h, :min_w]
    mu_x = float(img_a.mean())
    mu_y = float(img_b.mean())
    sigma_x = float(img_a.var())
    sigma_y = float(img_b.var())
    sigma_xy = float(((img_a - mu_x) * (img_b - mu_y)).mean())
    # Use small constants in luminance and contrast terms
    c1 = (0.01 * 1.0) ** 2
    c2 = (0.03 * 1.0) ** 2
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    if denominator == 0:
        return 0.0
    ssim = numerator / denominator
    return float(max(0.0, min(1.0, ssim)))


def compute_dssim_for_pdfs(pdf_a: str, pdf_b: str, dpi: int = 144) -> List[Dict[str, Any]]:
    """Compute DSSIM per page for two PDFs.

    Args:
        pdf_a: Path to the first PDF file.
        pdf_b: Path to the second PDF file.
        dpi: Rendering resolution in dots per inch.

    Returns:
        A list of dictionaries with fields ``page`` (1-based index), ``dssim``
        (the computed DSSIM value or ``None`` if a page is missing) and
        ``status`` describing the comparison result.
    """
    images_a = render_pdf_to_images(pdf_a, dpi=dpi)
    images_b = render_pdf_to_images(pdf_b, dpi=dpi)
    total_pages = max(len(images_a), len(images_b))
    results: List[Dict[str, Any]] = []
    for idx in range(total_pages):
        page_no = idx + 1
        if idx >= len(images_a) or idx >= len(images_b):
            results.append({"page": page_no, "dssim": None, "status": "missing"})
            continue
        ssim = structural_similarity(images_a[idx], images_b[idx])
        dssim = (1.0 - ssim) / 2.0
        results.append({"page": page_no, "dssim": float(dssim), "status": "ok"})
    return results