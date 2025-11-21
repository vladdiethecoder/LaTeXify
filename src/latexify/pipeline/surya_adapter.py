"""Surya layout detector wrapper used by the render-aware pipeline."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - heavy dependency
    from surya import SuryaLayoutModel  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    SuryaLayoutModel = None  # type: ignore


@dataclass
class SuryaRegion:
    label: str
    polygon: Sequence[Tuple[float, float]]
    bbox: Tuple[float, float, float, float]
    confidence: float

    def to_payload(self) -> Dict[str, object]:
        return {
            "label": self.label,
            "polygon": [list(point) for point in self.polygon],
            "bbox": [float(value) for value in self.bbox],
            "confidence": float(self.confidence),
        }


class SuryaLayoutDetector:
    """Thin wrapper that lazily loads Surya and emits normalized regions."""

    def __init__(self, checkpoint_dir: Path | None = None, enable_math_detector: bool = True) -> None:
        self._checkpoint_dir = checkpoint_dir
        self._enable_math_detector = enable_math_detector
        self._model = None

    def available(self) -> bool:
        return SuryaLayoutModel is not None

    def _ensure_model(self):  # pragma: no cover - heavy path
        if not self.available():
            raise RuntimeError("SuryaLayoutModel is unavailable; install surya-ocr to enable Surya backend.")
        if self._model is None:
            ckpt_dir = str(self._checkpoint_dir) if self._checkpoint_dir else None
            LOGGER.info("Loading Surya layout model (math detector=%s) from %s", self._enable_math_detector, ckpt_dir)
            self._model = SuryaLayoutModel(checkpoint_dir=ckpt_dir, enable_math=self._enable_math_detector)
        return self._model

    def detect(self, image_path: Path) -> List[SuryaRegion]:
        model = self._ensure_model()
        outputs = model.predict(image_path=str(image_path))
        regions: List[SuryaRegion] = []
        for item in outputs:
            polygon = item.get("polygon") or []
            bbox = item.get("bbox") or [0.0, 0.0, 0.0, 0.0]
            regions.append(
                SuryaRegion(
                    label=item.get("label", "unknown"),
                    polygon=tuple(tuple(point) for point in polygon),
                    bbox=tuple(float(v) for v in bbox),
                    confidence=float(item.get("score", 0.0)),
                )
            )
        return regions

    def export_regions(self, image_path: Path, output_json: Path) -> List[SuryaRegion]:
        regions = self.detect(image_path)
        payload = [region.to_payload() for region in regions]
        output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return regions


__all__ = ["SuryaLayoutDetector", "SuryaRegion"]
