"""Use a VLM to convert raster plots into TikZ code."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

from ..models.model_adapters import InternVLAdapter, InternVLConfig

PROMPT = (
    "You are a LaTeX graphics expert. Convert the plotted data in the image into "
    "TikZ/PGFPlots code. Include axis labels, legends, and data points approximated "
    "from the image. Respond ONLY with LaTeX code containing a complete tikzpicture."
)


def _sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).lower()


class GraphToTikZGenerator:
    """Converts raster graphs into TikZ code via InternVL."""

    def __init__(self) -> None:
        repo = os.environ.get("LATEXIFY_INTERNVL_MODEL", "OpenGVLab/InternVL3_5-8B")
        models_root = Path(
            os.environ.get(
                "LATEXIFY_MODELS_ROOT",
                str(Path(__file__).resolve().parents[2] / "models"),
            )
        ).expanduser()
        model_dir = models_root / "ocr" / _sanitize(repo)
        config = InternVLConfig(
            model_dir=model_dir,
            prompt=PROMPT,
            max_new_tokens=1024,
            temperature=0.0,
            top_p=0.1,
        )
        try:
            self.adapter = InternVLAdapter(config)
        except Exception:
            self.adapter = None

    def generate(self, image_path: Path) -> Optional[str]:
        if self.adapter is None or not image_path.exists():
            return None
        try:
            result = self.adapter.predict(image_path).strip()
        except Exception:
            return None
        if "\\begin{tikzpicture}" not in result:
            return None
        return result


__all__ = ["GraphToTikZGenerator"]
