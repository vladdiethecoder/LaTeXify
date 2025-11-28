#!/usr/bin/env python3
"""
Self-Optimization Loop for LaTeXify.

1. Run the release pipeline on a canonical sample PDF.
2. Perform visual regression between source and generated PDFs (SSIM-like score).
3. Run a lightweight professionalism check (page count, size).
4. If quality is below threshold, automatically tune prompt templates and rerun.
"""
import sys
import os
import subprocess
import logging
from pathlib import Path
from typing import Dict, Tuple

from omegaconf import OmegaConf

# Add project root to pythonpath for run_release.py imports
ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("self_optimizer")

SAMPLE_PDF = ROOT / "release/samples/sample.pdf"
OUTPUT_PDF = ROOT / "release/output/sample.pdf"
PROMPTS_YAML = ROOT / "src" / "latexify" / "prompts.yaml"


def _load_pdf_images(pdf_path: Path):
    """Rasterize PDF pages to NumPy arrays for visual comparison."""
    try:
        import fitz  # pymupdf
        import numpy as np  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        LOGGER.warning("Visual regression disabled (PyMuPDF/numpy missing): %s", exc)
        return None

    if not pdf_path.exists():
        LOGGER.warning("PDF %s does not exist for visual regression.", pdf_path)
        return None

    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        pix = page.get_pixmap(dpi=144)
        n_channels = 4 if pix.alpha else 3
        data = pix.samples
        arr = np.frombuffer(data, dtype="uint8")
        try:
            arr = arr.reshape(pix.height, pix.width, n_channels)[..., :3]
        except ValueError:
            # Fallback: skip malformed page
            LOGGER.warning("Failed to reshape pixmap for page %s; skipping.", page.number)
            continue
        pages.append(arr)
    return pages


def _page_similarity(a, b) -> float:
    """Compute a visual similarity between two RGB arrays (SSIM when available)."""
    import numpy as np  # type: ignore

    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    if h == 0 or w == 0:
        return 0.0
    a_crop = a[:h, :w, :].astype("float32")
    b_crop = b[:h, :w, :].astype("float32")

    # Prefer SSIM if skimage is installed.
    try:
        from skimage.metrics import structural_similarity  # type: ignore

        a_gray = (0.299 * a_crop[..., 0] + 0.587 * a_crop[..., 1] + 0.114 * a_crop[..., 2]).astype(
            "float32"
        )
        b_gray = (0.299 * b_crop[..., 0] + 0.587 * b_crop[..., 1] + 0.114 * b_crop[..., 2]).astype(
            "float32"
        )
        score = float(structural_similarity(a_gray, b_gray, data_range=255.0))
        return max(0.0, min(1.0, score))
    except Exception:
        # Fallback to a simple normalized MSE-based score.
        mse = float(((a_crop - b_crop) ** 2).mean())
        norm = 255.0 * 255.0
        mse_norm = mse / max(norm, 1.0)
        return max(0.0, 1.0 - mse_norm)


def visual_match_score() -> float:
    """Return a 0–1 visual similarity score between source and output PDFs."""
    ref_pages = _load_pdf_images(SAMPLE_PDF)
    out_pages = _load_pdf_images(OUTPUT_PDF)
    if not ref_pages or not out_pages:
        return 0.0
    pairs = min(len(ref_pages), len(out_pages))
    if pairs == 0:
        return 0.0
    scores = [_page_similarity(ref_pages[i], out_pages[i]) for i in range(pairs)]
    score = sum(scores) / len(scores)
    LOGGER.info("Visual regression score (SSIM-like): %.3f", score)
    return float(score)

def run_pipeline() -> bool:
    LOGGER.info(">>> Running LaTeXify Pipeline...")
    cmd = [
        "python", "run_release.py",
        "--pdf", str(SAMPLE_PDF),
        "--output-dir", "release/output",
        "--llm-device", "cpu",  # Use CPU by default in the loop
        "--pdf-dpi", "150",     # Faster loop
        "--chunk-size", "2000",
        "--enable-robust-compilation",
    ]
    # Allow operators to dial features down explicitly, but keep all
    # advanced stages enabled by default.
    if os.environ.get("LATEXIFY_SELFOPT_DISABLE_REFINEMENT", "0") == "1":
        cmd.append("--disable-refinement")
    try:
        subprocess.run(cmd, check=True, cwd=str(ROOT), timeout=1200)
    except subprocess.CalledProcessError as e:
        LOGGER.error("Pipeline failed: %s", e)
        return False
    return True

def evaluate_output() -> Tuple[float, Dict[str, float]]:
    """Compute a composite quality score and diagnostic breakdown."""
    LOGGER.info(">>> Evaluating Output...")
    diagnostics: Dict[str, float] = {}

    if not OUTPUT_PDF.exists():
        LOGGER.error("Output PDF missing.")
        diagnostics["visual_match"] = 0.0
        diagnostics["professional"] = 0.0
        return 0.0, diagnostics

    size = OUTPUT_PDF.stat().st_size
    diagnostics["size_bytes"] = float(size)
    if size < 5000:
        LOGGER.warning("PDF too small (%s bytes). Likely empty or failed compilation.", size)

    # Page count / basic professionalism proxy
    pages = 0
    try:
        import fitz  # type: ignore

        doc = fitz.open(OUTPUT_PDF)
        pages = len(doc)
        diagnostics["page_count"] = float(pages)
        LOGGER.info("Generated %s pages.", pages)
    except Exception as exc:  # pragma: no cover - optional dependency
        LOGGER.warning("PyMuPDF not available for page counting: %s", exc)

    # Visual regression score between source and generated PDFs
    vis_score = visual_match_score()
    diagnostics["visual_match"] = vis_score

    # Simple professionalism heuristic: >= 1 page, non-trivial size, decent visual score
    professional = 1.0 if (pages >= 1 and size > 5000 and vis_score >= 0.8) else 0.0
    diagnostics["professional"] = professional

    # Composite score: prioritize visual match, gated by professionalism
    composite = min(vis_score, 0.95 if professional else 0.85)
    LOGGER.info(
        "Composite quality score: %.3f (visual=%.3f, professional=%s)",
        composite,
        vis_score,
        "yes" if professional else "no",
    )
    return composite, diagnostics


def tune_prompts(iteration: int, diagnostics: Dict[str, float]) -> None:
    """
    Auto-adjust prompt templates based on evaluation diagnostics.

    Currently we enrich the refinement_prompt with stronger guidance toward
    high-fidelity, professional academic layout.
    """
    if not PROMPTS_YAML.exists():
        LOGGER.warning("Prompts file %s not found; skipping prompt tuning.", PROMPTS_YAML)
        return

    try:
        cfg = OmegaConf.load(str(PROMPTS_YAML))
    except Exception as exc:  # pragma: no cover - config guard
        LOGGER.warning("Failed to load prompts.yaml for tuning: %s", exc)
        return

    base_prompt = str(cfg.get("refinement_prompt") or "")
    marker = f"[auto-opt-{iteration}]"
    if marker in base_prompt:
        LOGGER.info("Prompts already tuned for iteration %d; skipping update.", iteration)
        return

    visual = diagnostics.get("visual_match", 0.0)
    professional_flag = diagnostics.get("professional", 0.0)
    augmentation = (
        f"\n\n% {marker} Visual score={visual:.3f}, professional={professional_flag:.0f}\n"
        "Emphasize visually faithful reproduction of the source PDF, including:\n"
        "- preserving page breaks and relative spacing where possible,\n"
        "- using consistent section headings and problem/answer boxes,\n"
        "- avoiding missing paragraphs or truncated equations."
    )
    cfg["refinement_prompt"] = base_prompt + augmentation
    try:
        OmegaConf.save(config=cfg, f=str(PROMPTS_YAML))
        LOGGER.info("Updated prompts.yaml refinement_prompt for iteration %d.", iteration)
    except Exception as exc:  # pragma: no cover - filesystem guard
        LOGGER.warning("Failed to save tuned prompts.yaml: %s", exc)


def main():
    LOGGER.info("Starting Self-Optimization Loop")
    
    max_iters = 3
    current_iter = 0
    best_score = 0.0
    last_score = 0.0
    
    while current_iter < max_iters:
        current_iter += 1
        LOGGER.info(f"=== Iteration {current_iter} ===")
        
        success = run_pipeline()
        if not success:
            LOGGER.error("Run failed. Aborting loop.")
            break

        score, diagnostics = evaluate_output()
        last_score = score
        best_score = max(best_score, score)
        LOGGER.info("Quality Score: %.3f", score)

        visual = diagnostics.get("visual_match", 0.0)
        professional = diagnostics.get("professional", 0.0) >= 1.0
        if score >= 0.95 and visual >= 0.95 and professional:
            LOGGER.info("Target quality reached. Optimization complete.")
            break

        LOGGER.info("Score below threshold. Tuning prompts for next iteration...")
        tune_prompts(current_iter, diagnostics)

    if current_iter >= max_iters and last_score < 0.95:
        LOGGER.warning("Max iterations reached without perfect score (best=%.3f).", best_score)

if __name__ == "__main__":
    main()
