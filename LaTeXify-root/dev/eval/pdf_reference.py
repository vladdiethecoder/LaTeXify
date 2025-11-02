# dev/eval/pdf_reference.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

<<<<<<< ours
import numpy as np
=======
>>>>>>> theirs
import fitz  # PyMuPDF
import numpy as np

try:  # Optional heavy deps
    from pdf2image import convert_from_path
    from pdf2image.exceptions import PDFInfoNotInstalledError, PDFPageCountError
except Exception:  # pragma: no cover - import guard
    convert_from_path = None
    PDFInfoNotInstalledError = PDFPageCountError = Exception

try:
    from PIL import Image
except Exception:  # pragma: no cover - import guard
    Image = None


@dataclass
class PDFImageComparison:
    reference: str
    candidate: str
    page_dssim: List[float]
    mean_dssim: float

@dataclass
class PageRef:
    text: str
    blocks: List[Dict[str, Any]]  # raw blocks with bbox and text

def extract_pdf_reference(pdf_path: str) -> List[PageRef]:
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
                "text": (b[4] or "").strip()
            })
        out.append(PageRef(text=text, blocks=blocks))
    return out


<<<<<<< ours
def _ensure_pdf2image() -> None:
    if convert_from_path is None or Image is None:
        raise RuntimeError("pdf2image and Pillow are required for PDF image comparison.")


def _render_pdf(path: Path, dpi: int) -> List[Any]:
    _ensure_pdf2image()
    try:
        return convert_from_path(str(path), dpi=dpi, fmt="png")
    except (PDFInfoNotInstalledError, PDFPageCountError) as exc:  # pragma: no cover - environment issue
        raise RuntimeError(f"pdf2image failed for {path}: {exc}")


def _to_gray_array(image: Any) -> np.ndarray:
    gray = image.convert("L") if hasattr(image, "convert") else image
    arr = np.asarray(gray, dtype=np.float32)
    return arr


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError("Arrays must share identical shape for SSIM computation.")
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    mu_a = a.mean()
    mu_b = b.mean()
    sigma_a = ((a - mu_a) ** 2).mean()
    sigma_b = ((b - mu_b) ** 2).mean()
    sigma_ab = ((a - mu_a) * (b - mu_b)).mean()
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    denom = (mu_a ** 2 + mu_b ** 2 + c1) * (sigma_a + sigma_b + c2)
    if denom == 0:
        return 1.0 if (mu_a == mu_b and sigma_a == sigma_b == 0) else 0.0
    return float(((2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)) / denom)


def compare_pdf_images(reference_pdf: Path, candidate_pdf: Path, dpi: int = 150) -> PDFImageComparison:
    """Render two PDFs and compute DSSIM per page (global SSIM heuristic)."""
    ref_imgs = _render_pdf(reference_pdf, dpi=dpi)
    cand_imgs = _render_pdf(candidate_pdf, dpi=dpi)

    page_scores: List[float] = []
    page_count = max(len(ref_imgs), len(cand_imgs))
    for idx in range(page_count):
        if idx >= len(ref_imgs) or idx >= len(cand_imgs):
            page_scores.append(1.0)
            continue
        ref_img = ref_imgs[idx]
        cand_img = cand_imgs[idx]
        if ref_img.size != cand_img.size:
            resample = getattr(Image, "BICUBIC", 3) if Image else 3
            cand_img = cand_img.resize(ref_img.size, resample=resample)
        ref_arr = _to_gray_array(ref_img)
        cand_arr = _to_gray_array(cand_img)
        ssim = _ssim(ref_arr, cand_arr)
        dssim = max(0.0, min(1.0, (1 - ssim) / 2))
        page_scores.append(float(dssim))

    mean_dssim = float(np.mean(page_scores)) if page_scores else 0.0
    return PDFImageComparison(
        reference=str(reference_pdf),
        candidate=str(candidate_pdf),
        page_dssim=page_scores,
        mean_dssim=mean_dssim,
    )
=======
def _pixmap_to_gray(pix: fitz.Pixmap) -> np.ndarray:
    if pix.alpha:  # drop alpha for consistent reshaping
        pix = fitz.Pixmap(fitz.csRGB, pix)
    data = np.frombuffer(pix.samples, dtype=np.uint8)
    arr = data.reshape(pix.height, pix.width, pix.n)
    if pix.n == 1:
        gray = arr[:, :, 0]
    else:
        rgb = arr[:, :, :3]
        gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    return (gray.astype(np.float32) / 255.0).copy()


def render_pdf_to_images(pdf_path: str, dpi: int = 144) -> List[np.ndarray]:
    doc = fitz.open(pdf_path)
    images: List[np.ndarray] = []
    for page in doc:
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        images.append(_pixmap_to_gray(pix))
    return images


def structural_similarity(img_a: np.ndarray, img_b: np.ndarray) -> float:
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
    c1 = (0.01 * 1.0) ** 2
    c2 = (0.03 * 1.0) ** 2
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    if denominator == 0:
        return 0.0
    ssim = numerator / denominator
    return float(max(0.0, min(1.0, ssim)))


def compute_dssim_for_pdfs(pdf_a: str, pdf_b: str, dpi: int = 144) -> List[Dict[str, Any]]:
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
>>>>>>> theirs
