"""Vision-oriented helpers that generate multi-view crops for downstream agents."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, MutableMapping, Sequence, Tuple

from PIL import Image, ImageEnhance, ImageOps

from ...core import common

try:  # pragma: no cover - torchvision optional in unit tests
    import torch
    from torchvision.transforms import functional as VF  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    VF = None  # type: ignore


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


@dataclass
class VisionSynthesisConfig:
    """Configuration for building multi-view document crops."""

    target_sizes: Tuple[int, ...] = (224, 320)
    max_views_per_chunk: int = 4
    pad_to_square: bool = True
    brightness: float = 0.1
    contrast: float = 0.15
    saturation: float = 0.05
    enable_grayscale: bool = True
    enable_tensor_views: bool = True
    normalize_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    normalize_std: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    background_color: Tuple[int, int, int] = (255, 255, 255)


@dataclass
class VisionView:
    """Persisted view metadata emitted by :class:`MultiViewRenderer`."""

    chunk_id: str
    view_index: int
    role: str
    path: Path
    size: Tuple[int, int]
    augmentation: str
    stats: Dict[str, float] = field(default_factory=dict)
    tensor: "torch.Tensor | None" = None

    def to_metadata(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "view_index": self.view_index,
            "role": self.role,
            "path": str(self.path),
            "size": list(self.size),
            "augmentation": self.augmentation,
            "stats": self.stats,
        }
        if self.tensor is not None:
            payload["tensor_shape"] = list(self.tensor.shape)
        return payload


@dataclass
class VisionSynthesisResult:
    """Container for the manifest returned by :class:`MultiViewRenderer`."""

    output_dir: Path
    views_by_chunk: Dict[str, List[VisionView]]

    def summary(self) -> Dict[str, object]:
        total = sum(len(views) for views in self.views_by_chunk.values())
        avg = total / max(1, len(self.views_by_chunk)) if self.views_by_chunk else 0.0
        return {
            "chunks": len(self.views_by_chunk),
            "views": total,
            "avg_views_per_chunk": round(avg, 2),
        }

    def attach_metadata(self, chunks: Sequence[common.Chunk]) -> List[common.Chunk]:
        updated: List[common.Chunk] = []
        for chunk in chunks:
            metadata = dict(chunk.metadata or {})
            views = self.views_by_chunk.get(chunk.chunk_id)
            if views:
                metadata["vision_views"] = [view.to_metadata() for view in views]
            updated.append(
                common.Chunk(
                    chunk_id=chunk.chunk_id,
                    page=chunk.page,
                    text=chunk.text,
                    images=list(chunk.images),
                    metadata=metadata,
                )
            )
        return updated


@dataclass
class _ViewSpec:
    image: Image.Image
    role: str
    augmentation: str
    score: float
    size: Tuple[int, int]
    metadata: Dict[str, float]
    tensor: "torch.Tensor | None" = None


class ImageViewGenerator:
    """Generate normalized/augmented crops for a single chunk."""

    def __init__(self, config: VisionSynthesisConfig | None = None) -> None:
        self.config = config or VisionSynthesisConfig()

    def generate(self, chunk: common.Chunk) -> List[_ViewSpec]:
        sources: List[Tuple[Image.Image, str]] = []
        for idx, image_path in enumerate(chunk.images):
            image = self._load_image(Path(image_path))
            if image:
                sources.append((image, f"chunk-image-{idx}"))
        if not sources:
            crop = self._crop_from_metadata(chunk.metadata or {})
            if crop:
                sources.append((crop, "page-crop"))
        specs: List[_ViewSpec] = []
        for source_idx, (image, role) in enumerate(sources):
            specs.extend(self._augment_views(image, role, chunk.chunk_id, source_idx))
            if len(specs) >= self.config.max_views_per_chunk:
                break
        return specs[: self.config.max_views_per_chunk]

    def _load_image(self, path: Path) -> Image.Image | None:
        if not path.exists():
            return None
        try:
            with Image.open(path) as handle:
                return handle.convert("RGB")
        except Exception:
            return None

    def _crop_from_metadata(self, metadata: MutableMapping[str, object]) -> Image.Image | None:
        page_image = metadata.get("page_image")
        bbox = metadata.get("bbox")
        page_width = metadata.get("page_width_pt") or metadata.get("page_width")
        page_height = metadata.get("page_height_pt") or metadata.get("page_height")
        if not page_image or not bbox or page_width in (None, 0) or page_height in (None, 0):
            return None
        try:
            coords = tuple(float(value) for value in bbox)
        except Exception:
            return None
        path = Path(str(page_image))
        if not path.exists():
            return None
        try:
            with Image.open(path) as handle:
                width_px, height_px = handle.size
                scale_x = width_px / max(float(page_width), 1e-3)
                scale_y = height_px / max(float(page_height), 1e-3)
                x0, y0, x1, y1 = coords
                crop_box = (
                    int(max(0, min(width_px, x0 * scale_x))),
                    int(max(0, min(height_px, y0 * scale_y))),
                    int(max(0, min(width_px, x1 * scale_x))),
                    int(max(0, min(height_px, y1 * scale_y))),
                )
                if crop_box[2] - crop_box[0] < 2 or crop_box[3] - crop_box[1] < 2:
                    return None
                margin = max(4, int(0.01 * max(width_px, height_px)))
                expanded = (
                    max(0, crop_box[0] - margin),
                    max(0, crop_box[1] - margin),
                    min(width_px, crop_box[2] + margin),
                    min(height_px, crop_box[3] + margin),
                )
                return handle.crop(expanded).convert("RGB")
        except Exception:
            return None

    def _augment_views(self, image: Image.Image, role: str, chunk_id: str, source_idx: int) -> List[_ViewSpec]:
        views: List[_ViewSpec] = []
        base = self._normalize(image)
        for size in self.config.target_sizes:
            resized = self._resize(base, size)
            tensor = self._to_tensor(resized)
            views.append(
                _ViewSpec(
                    image=resized,
                    role=role,
                    augmentation=f"base-{size}",
                    score=self._sharpness(resized),
                    size=resized.size,
                    metadata={"source_index": float(source_idx)},
                    tensor=tensor,
                )
            )
            if self.config.enable_grayscale:
                gray = ImageOps.grayscale(resized).convert("RGB")
                tensor = self._to_tensor(gray)
                views.append(
                    _ViewSpec(
                        image=gray,
                        role=role,
                        augmentation=f"grayscale-{size}",
                        score=self._sharpness(gray),
                        size=gray.size,
                        metadata={"source_index": float(source_idx), "grayscale": 1.0},
                        tensor=tensor,
                    )
                )
            jittered = self._color_jitter(resized)
            if jittered is not None:
                tensor = self._to_tensor(jittered)
                views.append(
                    _ViewSpec(
                        image=jittered,
                        role=role,
                        augmentation=f"jitter-{size}",
                        score=self._sharpness(jittered),
                        size=jittered.size,
                        metadata={"source_index": float(source_idx), "jitter": 1.0},
                        tensor=tensor,
                    )
                )
            if len(views) >= self.config.max_views_per_chunk:
                break
        return views

    def _normalize(self, image: Image.Image) -> Image.Image:
        if not self.config.pad_to_square:
            return image
        width, height = image.size
        size = max(width, height)
        canvas = Image.new("RGB", (size, size), self.config.background_color)
        offset = ((size - width) // 2, (size - height) // 2)
        canvas.paste(image, offset)
        return canvas

    def _resize(self, image: Image.Image, target: int) -> Image.Image:
        if target <= 0:
            return image
        return image.resize((target, target), Image.BICUBIC)

    def _color_jitter(self, image: Image.Image) -> Image.Image | None:
        if self.config.brightness <= 0 and self.config.contrast <= 0 and self.config.saturation <= 0:
            return None
        jittered = image
        if self.config.brightness > 0:
            factor = 1.0 + random.uniform(-self.config.brightness, self.config.brightness)
            jittered = ImageEnhance.Brightness(jittered).enhance(_clamp(factor, 0.1, 1.9))
        if self.config.contrast > 0:
            factor = 1.0 + random.uniform(-self.config.contrast, self.config.contrast)
            jittered = ImageEnhance.Contrast(jittered).enhance(_clamp(factor, 0.1, 1.9))
        if self.config.saturation > 0:
            factor = 1.0 + random.uniform(-self.config.saturation, self.config.saturation)
            jittered = ImageEnhance.Color(jittered).enhance(_clamp(factor, 0.1, 1.9))
        return jittered

    def _sharpness(self, image: Image.Image) -> float:
        gray = ImageOps.grayscale(image)
        histogram = gray.histogram()
        total = sum(histogram)
        if total == 0:
            return 0.0
        mean = sum(idx * value for idx, value in enumerate(histogram)) / total
        variance = sum(value * ((idx - mean) ** 2) for idx, value in enumerate(histogram)) / total
        return round(_clamp(math.sqrt(variance) / 128.0), 3)

    def _to_tensor(self, image: Image.Image) -> "torch.Tensor | None":
        if not self.config.enable_tensor_views or torch is None or VF is None:
            return None
        tensor = VF.to_tensor(image)
        mean = torch.tensor(self.config.normalize_mean).view(-1, 1, 1)
        std = torch.tensor(self.config.normalize_std).view(-1, 1, 1)
        normalized = (tensor - mean) / std
        return normalized


class MultiViewRenderer:
    """Drive :class:`ImageViewGenerator` over a batch of chunks."""

    def __init__(self, config: VisionSynthesisConfig | None = None, *, output_dir: Path | None = None) -> None:
        self.config = config or VisionSynthesisConfig()
        self.generator = ImageViewGenerator(self.config)
        self._output_dir = output_dir

    def render(
        self,
        chunks: Iterable[common.Chunk],
        *,
        output_dir: Path | None = None,
    ) -> VisionSynthesisResult:
        target_dir = output_dir or self._output_dir
        if target_dir is None:
            raise ValueError("output_dir must be provided when calling render()")
        target_dir.mkdir(parents=True, exist_ok=True)
        views_by_chunk: Dict[str, List[VisionView]] = {}
        for chunk in chunks:
            specs = self.generator.generate(chunk)
            if not specs:
                continue
            chunk_views: List[VisionView] = []
            for view_index, spec in enumerate(specs):
                file_name = f"{chunk.chunk_id}_v{view_index}_{spec.augmentation}.png"
                view_path = target_dir / file_name
                spec.image.save(view_path)
                chunk_views.append(
                    VisionView(
                        chunk_id=chunk.chunk_id,
                        view_index=view_index,
                        role=spec.role,
                        path=view_path,
                        size=spec.size,
                        augmentation=spec.augmentation,
                        stats={"score": spec.score, **spec.metadata},
                        tensor=spec.tensor,
                    )
                )
            views_by_chunk[chunk.chunk_id] = chunk_views
        return VisionSynthesisResult(output_dir=target_dir, views_by_chunk=views_by_chunk)


__all__ = [
    "ImageViewGenerator",
    "MultiViewRenderer",
    "VisionSynthesisConfig",
    "VisionSynthesisResult",
    "VisionView",
]
