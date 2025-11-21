"""Resolve ambiguous OCR characters by re-reading regions with a VLM."""
from __future__ import annotations

import contextlib
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Dict, Tuple

from PIL import Image

from ..models.model_adapters import InternVLAdapter, InternVLConfig

AMBIG_PATTERN = re.compile(r"(?i)(\b[S5]0[B8]|\bSOB\b|[Il1]{2,}|0O|O0)")
PROMPT = (
    "Transcribe the highlighted snippet exactly as typeset (case sensitive, distinguish l vs 1). "
    "Return plain text only."
)

LOGGER = logging.getLogger(__name__)


class AmbiguityResolver:
    """Uses InternVL to clarify ambiguous glyphs."""

    def __init__(self) -> None:
        repo = os.environ.get("LATEXIFY_AMBIGUITY_VLM", os.environ.get("LATEXIFY_INTERNVL_MODEL", "OpenGVLab/InternVL3_5-8B"))
        model_root = Path(
            os.environ.get("LATEXIFY_MODELS_ROOT", str(Path(__file__).resolve().parents[2] / "models"))
        ).expanduser()
        target = model_root / "ocr" / re.sub(r"[^A-Za-z0-9._-]+", "-", repo).lower()
        self._config = InternVLConfig(
            model_dir=target,
            prompt=PROMPT,
            max_new_tokens=512,
            temperature=0.0,
            top_p=0.1,
        )
        self.adapter: InternVLAdapter | None = None

    def _ensure_adapter(self) -> bool:
        if self.adapter is not None:
            return True
        try:
            self.adapter = InternVLAdapter(self._config)
        except Exception as exc:
            LOGGER.warning("Ambiguity resolver could not load InternVL: %s", exc)
            self.adapter = None
        return self.adapter is not None

    def _has_ambiguity(self, text: str) -> bool:
        return bool(AMBIG_PATTERN.search(text))

    @staticmethod
    def _crop_region(metadata: Dict[str, object]) -> Tuple[Path | None, Path | None]:
        bbox = metadata.get("bbox")
        page_image = metadata.get("page_image")
        page_width = metadata.get("page_width_pt") or metadata.get("page_width")
        page_height = metadata.get("page_height_pt") or metadata.get("page_height")
        if not bbox or not page_image or page_width in (None, 0) or page_height in (None, 0):
            return None, None
        try:
            bbox_tuple = tuple(float(coord) for coord in bbox)
        except Exception:
            return None, None
        path = Path(page_image)
        if not path.exists():
            return None, None
        try:
            with Image.open(path) as image:
                width_px, height_px = image.size
                scale_x = width_px / max(float(page_width), 1e-3)
                scale_y = height_px / max(float(page_height), 1e-3)
                x0, y0, x1, y1 = bbox_tuple
                left = int(max(0, min(width_px, x0 * scale_x)))
                top = int(max(0, min(height_px, y0 * scale_y)))
                right = int(max(left + 1, min(width_px, x1 * scale_x)))
                bottom = int(max(top + 1, min(height_px, y1 * scale_y)))
                margin = max(4, int(0.01 * max(width_px, height_px)))
                left = max(0, left - margin)
                right = min(width_px, right + margin)
                top = max(0, top - margin)
                bottom = min(height_px, bottom + margin)
                if left >= right or top >= bottom:
                    return None, None
                crop = image.crop((left, top, right, bottom))
                temp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                crop.save(temp.name)
                temp.close()
                return Path(temp.name), path
        except Exception:
            return None, None

    def maybe_fix(self, chunk) -> bool:
        if not self._ensure_adapter():
            return False
        text = chunk.text or ""
        if not self._has_ambiguity(text):
            return False
        crop_path, _ = self._crop_region(chunk.metadata or {})
        if crop_path is None:
            return False
        try:
            suggestion = self.adapter.predict(crop_path)
        finally:
            try:
                crop_path.unlink()
            except Exception:
                pass
        if not suggestion:
            return False
        suggestion = suggestion.strip()
        if not suggestion:
            return False
        chunk.text = suggestion
        return True

    def close(self) -> None:
        if self.adapter is None:
            return
        torch = getattr(self.adapter, "torch", None)
        try:
            if hasattr(self.adapter, "model"):
                self.adapter.model.to("cpu")
        except Exception:
            LOGGER.debug("Unable to move ambiguity resolver adapter to CPU.", exc_info=True)
        if torch is not None:
            if getattr(torch, "cuda", None) and torch.cuda.is_available():
                with contextlib.suppress(Exception):
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
        self.adapter = None
