"""Multimodal reward helpers backed by InternVL."""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

try:  # pragma: no cover - optional dependency
    from pdf2image import convert_from_path
except Exception:  # pragma: no cover
    convert_from_path = None  # type: ignore

from ..core import common
from ..models.model_adapters import (
    InternVLAdapter,
    InternVLConfig,
    get_shared_adapter,
    register_shared_adapter,
    release_shared_adapter,
)

LOGGER = logging.getLogger(__name__)
REWARD_PROMPT = (
    "You are a meticulous LaTeX layout judge. Given the rendered page, rate its "
    "visual polish, readability, and mathematical hygiene.\n\n"
    "Respond with a single line of the form:\nscore: <decimal between 0 and 1>\n"
    "Optionally append a short justification after the score."
)
ROOT = Path(__file__).resolve().parents[2]
def _sanitize_model_subdir(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).lower()


INTERNVL_MODEL_ID = os.environ.get("LATEXIFY_INTERNVL_MODEL", "OpenGVLab/InternVL3_5-8B")
DEFAULT_INTERNVL_DIR = ROOT / "models" / "ocr" / _sanitize_model_subdir(INTERNVL_MODEL_ID)
_SCORER: "InternVLReward" | None = None


@dataclass
class InternVLReward:
    """Thin wrapper that reuses InternVL for qualitative scoring."""

    model_dir: Path
    max_new_tokens: int = 96
    adapter: InternVLAdapter | None = None

    def __post_init__(self) -> None:
        if self.adapter is not None:
            return
        config = InternVLConfig(
            model_dir=self.model_dir,
            max_new_tokens=self.max_new_tokens,
            prompt=REWARD_PROMPT,
            temperature=0.0,
            top_p=0.1,
        )
        self.adapter = InternVLAdapter(config)

    def score_image(self, image_path: Path) -> float:
        if self.adapter is None:
            raise RuntimeError("InternVL reward adapter is not initialized.")
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


def _render_pdf_previews(tex_path: Path, pages: Sequence[int] | None, dpi: int) -> List[Path]:
    pdf_path = tex_path.with_suffix(".pdf")
    if not pdf_path.exists():
        LOGGER.debug("PDF not found for multimodal reward: %s", pdf_path)
        return []
    if convert_from_path is None:
        LOGGER.warning("pdf2image is unavailable; cannot render PDF preview for mm reward.")
        return []
    targets = sorted({page for page in (pages or [1]) if page and page > 0})
    if not targets:
        targets = [1]
    previews: List[Path] = []
    for page in targets:
        try:
            images = convert_from_path(
                str(pdf_path),
                first_page=page,
                last_page=page,
                fmt="png",
                dpi=max(72, dpi),
            )
        except Exception as exc:  # pragma: no cover - depends on poppler availability
            LOGGER.warning("Failed to rasterize %s page %s for mm reward: %s", pdf_path, page, exc)
            continue
        if not images:
            continue
        preview_path = tex_path.with_suffix(f".mmreward_{page:04d}.png")
        try:
            images[0].save(preview_path)
        except Exception as exc:
            LOGGER.warning("Unable to save preview image %s: %s", preview_path, exc)
            continue
        previews.append(preview_path)
    return previews


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
    shared_adapter = None
    try:
        candidate = get_shared_adapter("internvl")
        if isinstance(candidate, InternVLAdapter):
            shared_adapter = candidate
    except Exception:
        shared_adapter = None
    _SCORER = InternVLReward(model_dir=model_root, adapter=shared_adapter)
    if shared_adapter is None and _SCORER.adapter is not None:
        register_shared_adapter("internvl", _SCORER.adapter)
    return _SCORER


def aesthetic_mm_score(tex_path: Path, pages: Sequence[int] | None = None, dpi: int = 150) -> float:
    """Score aesthetics via InternVL on the compiled PDF preview."""

    previews = _render_pdf_previews(tex_path, pages, dpi)
    if not previews:
        return 0.0
    try:
        scorer = _get_scorer()
        total = 0.0
        for preview in previews:
            total += scorer.score_image(preview)
        score = total / len(previews) if previews else 0.0
    finally:
        for preview in previews:
            try:
                preview.unlink()
            except FileNotFoundError:
                continue
    return score


def visual_textual_consistency(
    tex_path: Path,
    plan_path: Path,
    chunks_path: Path,
    output_path: Path | None = None,
    assets_dir: Path | None = None,
) -> Dict[str, object]:
    """Generate figure/equation/layout consistency report using multimodal cues."""

    from .consistency_verifier import ConsistencyVerifier

    plan = common.load_plan(plan_path) if plan_path.exists() else []
    chunk_map = {chunk.chunk_id: chunk for chunk in common.load_chunks(chunks_path)} if chunks_path.exists() else {}
    verifier = ConsistencyVerifier()
    assets = assets_dir or tex_path.parent / "assets"
    report = verifier.evaluate(tex_path, plan, chunk_map, assets)
    payload = report.to_json()
    target = output_path or tex_path.parent / "consistency.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def release_reward_adapter() -> None:
    """Release references so InternVL VRAM can be reclaimed after scoring."""

    global _SCORER
    _SCORER = None
    try:
        release_shared_adapter("internvl")
    except Exception:
        LOGGER.debug("Failed to release shared InternVL adapter", exc_info=True)


__all__ = ["aesthetic_mm_score", "visual_textual_consistency", "release_reward_adapter"]
