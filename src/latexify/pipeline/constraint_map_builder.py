"""Compose symbolic constraint maps for render-aware reconstruction."""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

try:  # pragma: no cover - optional dependency
    from reportlab.graphics.shapes import Drawing, Image as RLImage, Rect  # type: ignore
    from reportlab.graphics import renderPM  # type: ignore
    from reportlab.lib import colors  # type: ignore
except Exception:  # pragma: no cover
    Drawing = None  # type: ignore
    RLImage = None  # type: ignore
    Rect = None  # type: ignore
    renderPM = None  # type: ignore
    colors = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from PIL import Image as PILImage, ImageDraw
except Exception:  # pragma: no cover
    PILImage = None  # type: ignore
    ImageDraw = None  # type: ignore

from .symbolic_render import FormulaRenderer

LOGGER = logging.getLogger(__name__)


@dataclass
class ConstraintMapArtifact:
    page_index: int
    constraint_map: Path
    mask_path: Path
    rendered_items: int


class ConstraintMapBuilder:
    FORMULA_TAGS = {"formula", "equation", "math"}
    FIGURE_TAGS = {"figure", "image", "table"}

    def __init__(self, renderer: FormulaRenderer, *, page_images_dir: Path | None = None) -> None:
        if Drawing is None or renderPM is None or Rect is None:
            raise ImportError("reportlab is required to build constraint maps. Install `reportlab`. ")
        if PILImage is None or ImageDraw is None:
            raise ImportError("Pillow is required to build constraint maps. Install `Pillow`. ")
        self.renderer = renderer
        self.page_images_dir = page_images_dir

    def build_from_master_items(
        self,
        master_items_path: Path,
        output_dir: Path,
        allowed_pages: Iterable[int] | None = None,
    ) -> List[ConstraintMapArtifact]:
        output_dir.mkdir(parents=True, exist_ok=True)
        pages_filter = set(int(p) for p in allowed_pages) if allowed_pages else None
        payload = json.loads(master_items_path.read_text(encoding="utf-8"))
        grouped: Dict[int, List[Dict[str, object]]] = {}
        for entry in payload:
            page = int(entry.get("page", 1))
            if pages_filter and page not in pages_filter:
                continue
            grouped.setdefault(page, []).append(entry)
        artifacts: List[ConstraintMapArtifact] = []
        for page_index in sorted(grouped):
            artifact = self._build_page(page_index, grouped[page_index], output_dir)
            if artifact:
                artifacts.append(artifact)
        return artifacts

    def _build_page(
        self,
        page_index: int,
        items: List[Dict[str, object]],
        output_dir: Path,
    ) -> ConstraintMapArtifact | None:
        page_width, page_height = self._page_dimensions(items)
        drawing = Drawing(page_width, page_height)
        drawing.add(Rect(0, 0, page_width, page_height, fillColor=colors.whitesmoke, strokeColor=colors.whitesmoke))
        mask_image = PILImage.new(
            "L",
            (max(1, int(math.ceil(page_width))), max(1, int(math.ceil(page_height)))),
            color=255,
        )
        mask_draw = ImageDraw.Draw(mask_image)
        page_image = self._load_page_image(page_index)
        placed = 0
        for idx, entry in enumerate(items):
            bbox = self._normalize_bbox(entry.get("bbox"), page_width, page_height)
            if bbox is None:
                continue
            region_type = str(entry.get("region_type", "text")).lower()
            if region_type in self.FORMULA_TAGS and entry.get("content"):
                rendered = self.renderer.render_formula(str(entry["content"]))
                if self._place_image(drawing, bbox, page_height, rendered.image_path):
                    self._protect_mask(mask_draw, bbox, page_height)
                    placed += 1
            elif region_type in self.FIGURE_TAGS and page_image is not None:
                crop_path = self._extract_crop(
                    page_index,
                    idx,
                    bbox,
                    page_width,
                    page_height,
                    page_image,
                    output_dir,
                )
                if crop_path and self._place_image(drawing, bbox, page_height, crop_path):
                    self._protect_mask(mask_draw, bbox, page_height)
                    placed += 1
        constraint_path = output_dir / f"page_{page_index:04d}_constraint.png"
        renderPM.drawToFile(drawing, str(constraint_path), fmt="PNG")
        mask_path = output_dir / f"page_{page_index:04d}_mask.png"
        mask_image.save(mask_path)
        if page_image is not None:
            page_image.close()
        if placed == 0:
            LOGGER.info("No renderable regions on page %s; blank constraint map emitted.", page_index)
        return ConstraintMapArtifact(
            page_index=page_index,
            constraint_map=constraint_path,
            mask_path=mask_path,
            rendered_items=placed,
        )

    @staticmethod
    def _page_dimensions(items: List[Dict[str, object]]) -> Tuple[float, float]:
        for entry in items:
            width = entry.get("page_width_pt") or entry.get("page_width")
            height = entry.get("page_height_pt") or entry.get("page_height")
            if width and height:
                return float(width), float(height)
        return 612.0, 792.0  # Letter fallback

    @staticmethod
    def _normalize_bbox(bbox_value, page_width: float, page_height: float) -> Tuple[float, float, float, float] | None:
        if not bbox_value:
            return None
        try:
            x0, y0, x1, y1 = [float(v) for v in bbox_value]
        except Exception:
            return None
        x0 = max(0.0, min(page_width, x0))
        x1 = max(x0, min(page_width, x1))
        y0 = max(0.0, min(page_height, y0))
        y1 = max(y0, min(page_height, y1))
        if x1 - x0 <= 1e-2 or y1 - y0 <= 1e-2:
            return None
        return (x0, y0, x1, y1)

    @staticmethod
    def _place_image(drawing: Drawing, bbox: Tuple[float, float, float, float], page_height: float, image_path: Path) -> bool:
        try:
            x0, y0, width, height = ConstraintMapBuilder._canvas_rect(bbox, page_height)
        except ValueError:
            return False
        drawing.add(RLImage(x0, y0, width, height, str(image_path)))
        return True

    @staticmethod
    def _canvas_rect(bbox: Tuple[float, float, float, float], page_height: float) -> Tuple[float, float, float, float]:
        x0, y0, x1, y1 = bbox
        width = x1 - x0
        height = y1 - y0
        y_bottom = page_height - y1
        if width <= 0 or height <= 0:
            raise ValueError("Invalid bounding box dimensions.")
        return x0, y_bottom, width, height

    @staticmethod
    def _protect_mask(draw: ImageDraw.ImageDraw, bbox: Tuple[float, float, float, float], _: float) -> None:
        x0, y0, x1, y1 = bbox
        draw.rectangle([x0, y0, x1, y1], fill=0)

    def _load_page_image(self, page_index: int):
        if self.page_images_dir is None:
            return None
        candidate = self.page_images_dir / f"page_{page_index:04d}.png"
        if not candidate.exists():
            return None
        try:
            return PILImage.open(candidate)
        except Exception:
            LOGGER.debug("Failed to open page raster %s", candidate, exc_info=True)
            return None

    def _extract_crop(
        self,
        page_index: int,
        region_idx: int,
        bbox: Tuple[float, float, float, float],
        page_width: float,
        page_height: float,
        page_image,
        output_dir: Path,
    ) -> Path | None:
        if page_image is None:
            return None
        img_width, img_height = page_image.size
        scale_x = img_width / max(page_width, 1)
        scale_y = img_height / max(page_height, 1)
        x0, y0, x1, y1 = bbox
        top = max(0, int(y0 * scale_y))
        bottom = max(0, int(y1 * scale_y))
        left = max(0, int(x0 * scale_x))
        right = max(0, int(x1 * scale_x))
        if right - left <= 0 or bottom - top <= 0:
            return None
        crop = page_image.crop((left, top, right, bottom))
        crop_path = output_dir / f"page_{page_index:04d}_region{region_idx:03d}_crop.png"
        crop.save(crop_path)
        return crop_path


__all__ = ["ConstraintMapBuilder", "ConstraintMapArtifact"]
