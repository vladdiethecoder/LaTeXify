"""Deterministic formula renderer used by the render-aware pipeline."""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

try:  # pragma: no cover - optional dependency
    from matplotlib import mathtext  # type: ignore
except Exception:  # pragma: no cover
    mathtext = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore

LOGGER = logging.getLogger(__name__)


@dataclass
class RenderedFormula:
    image_path: Path
    width: int
    height: int
    baseline_offset: float


class FormulaRenderer:
    """Render LaTeX/mathtext strings into cached PNGs suitable for compositing."""

    def __init__(self, cache_dir: Path, dpi: int = 240, color: str = "#000000") -> None:
        if mathtext is None:
            raise ImportError("matplotlib is required to enable formula rendering. Install `matplotlib`. ")
        if Image is None:
            raise ImportError("Pillow is required for formula rendering. Install `Pillow`. ")
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.color = color
        self._parser = mathtext.MathTextParser("path")

    def render_formula(self, formula: str, fontsize: int = 16) -> RenderedFormula:
        sanitized = self._prepare_formula(formula)
        cache_key = hashlib.sha1(f"{sanitized}|{fontsize}|{self.dpi}".encode("utf-8")).hexdigest()
        target = self.cache_dir / f"{cache_key}.png"
        if not target.exists():
            self._parser.to_png(
                str(target),
                sanitized,
                dpi=self.dpi,
                color=self.color,
                fontsize=fontsize,
            )
        with Image.open(target) as img:
            width, height = img.size
        # Approximate baseline offset using a simple fraction of the rendered height.
        baseline_offset = max(1.0, height * 0.2)
        return RenderedFormula(image_path=target, width=width, height=height, baseline_offset=baseline_offset)

    @staticmethod
    def _prepare_formula(value: str) -> str:
        text = value.strip() or r"\text{ }"
        if not text.startswith("$"):
            text = f"${text}$"
        return text


__all__ = ["FormulaRenderer", "RenderedFormula"]
