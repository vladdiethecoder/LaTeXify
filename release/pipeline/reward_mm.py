"""Multimodal reward helpers backed by InternVL."""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:  # pragma: no cover - optional dependency
    from pdf2image import convert_from_path
except Exception:  # pragma: no cover
    convert_from_path = None  # type: ignore

from ..models.model_adapters import InternVLAdapter, InternVLConfig

LOGGER = logging.getLogger(__name__)
REWARD_PROMPT = (
    "You are a meticulous LaTeX layout judge. Given the rendered page, rate its "
    "visual polish, readability, and mathematical hygiene.\n\n"
    "Respond with a single line of the form:\nscore: <decimal between 0 and 1>\n"
    "Optionally append a short justification after the score."
)
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INTERNVL_DIR = ROOT / "models" / "ocr" / "internvl-3.5-14b"
_SCORER: "InternVLReward" | None = None


@dataclass
class InternVLReward:
    """Thin wrapper that reuses InternVL for qualitative scoring."""

    model_dir: Path
    max_new_tokens: int = 96

    def __post_init__(self) -> None:
        config = InternVLConfig(
            model_dir=self.model_dir,
            max_new_tokens=self.max_new_tokens,
            prompt=REWARD_PROMPT,
            temperature=0.0,
            top_p=0.1,
        )
        self.adapter = InternVLAdapter(config)

    def score_image(self, image_path: Path) -> float:
        response = self.adapter.predict(image_path)
        return _extract_score(response)


def _extract_score(response: str) -> float:
    match = re.search(r"([-+]?\d*\.\d+|\d+)", response)
    if not match:
        LOGGER.debug("Reward backend returned unparseable response: %s", response)
        return 0.0
    try:
        value = float(match.group(0))
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, value))


def _render_pdf_preview(tex_path: Path) -> Optional[Path]:
    pdf_path = tex_path.with_suffix(".pdf")
    if not pdf_path.exists():
        LOGGER.debug("PDF not found for multimodal reward: %s", pdf_path)
        return None
    if convert_from_path is None:
        LOGGER.warning("pdf2image is unavailable; cannot render PDF preview for mm reward.")
        return None
    try:
        images = convert_from_path(
            str(pdf_path),
            first_page=1,
            last_page=1,
            fmt="png",
            dpi=150,
        )
    except Exception as exc:  # pragma: no cover - depends on poppler availability
        LOGGER.warning("Failed to rasterize %s for mm reward: %s", pdf_path, exc)
        return None
    if not images:
        return None
    preview_path = tex_path.with_suffix(".mmreward.png")
    try:
        images[0].save(preview_path)
    except Exception as exc:
        LOGGER.warning("Unable to save preview image %s: %s", preview_path, exc)
        return None
    return preview_path


def _get_scorer() -> InternVLReward:
    global _SCORER
    if _SCORER is not None:
        return _SCORER
    model_root = Path(
        os.environ.get("LATEXIFY_MM_REWARD_MODEL_DIR", DEFAULT_INTERNVL_DIR)
    ).expanduser()
    if not model_root.exists():
        raise FileNotFoundError(
            f"Multimodal reward model directory missing: {model_root}. "
            "Please run the release bootstrapper or set LATEXIFY_MM_REWARD_MODEL_DIR."
        )
    _SCORER = InternVLReward(model_dir=model_root)
    return _SCORER


def aesthetic_mm_score(tex_path: Path) -> float:
    """Score aesthetics via InternVL on the compiled PDF preview."""

    preview = _render_pdf_preview(tex_path)
    if preview is None:
        return 0.0
    try:
        scorer = _get_scorer()
        score = scorer.score_image(preview)
    finally:
        try:
            preview.unlink()
        except FileNotFoundError:
            pass
    return score


__all__ = ["aesthetic_mm_score"]
