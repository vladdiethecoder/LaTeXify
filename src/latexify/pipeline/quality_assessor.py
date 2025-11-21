"""Input document quality assessment used to steer ingestion intensity."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Sequence

from PIL import Image

from ..core import common


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


@dataclass
class QualityProfile:
    """Compact representation of document legibility and structure."""

    ocr_confidence: float
    image_quality: float
    structure_clarity: float
    aggregate: float
    tier: str
    processing_mode: str
    vision_signal: float
    branch_consistency: float
    branch_strategy: str
    notes: Dict[str, object]

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["ocr_confidence"] = round(self.ocr_confidence, 3)
        payload["image_quality"] = round(self.image_quality, 3)
        payload["structure_clarity"] = round(self.structure_clarity, 3)
        payload["aggregate"] = round(self.aggregate, 3)
        payload["vision_signal"] = round(self.vision_signal, 3)
        payload["branch_consistency"] = round(self.branch_consistency, 3)
        return payload

    def attach_branch_metrics(self, metrics: Dict[str, float] | None) -> None:
        if not metrics:
            return
        branch_metrics = {
            key: round(float(value), 3)
            for key, value in metrics.items()
            if isinstance(value, (int, float))
        }
        if not branch_metrics:
            return
        existing = self.notes.get("branch_metrics")
        if isinstance(existing, dict):
            existing.update(branch_metrics)
        else:
            self.notes["branch_metrics"] = branch_metrics


class InputQualityAssessor:
    """Estimate input quality before and after OCR so downstream stages can adapt."""

    def __init__(self, poor_threshold: float = 0.45, excellent_threshold: float = 0.75) -> None:
        self.poor_threshold = poor_threshold
        self.excellent_threshold = excellent_threshold

    # ---------------------------- preview phase ---------------------------- #
    def preview_from_pages(self, pages: Sequence[str], figures: Sequence[Path]) -> QualityProfile:
        text_lengths = [len(page.strip()) for page in pages]
        avg_chars = mean(text_lengths) if text_lengths else 0.0
        text_density = _clamp(avg_chars / 900.0)
        figure_pressure = len(figures) / max(1, len(pages))
        image_quality = _clamp(1.0 - figure_pressure * 0.4)
        structure_hits = sum(1 for page in pages if self._looks_structured(page))
        structure_clarity = _clamp(structure_hits / max(1, len(pages)))
        aggregate = mean([text_density, image_quality, structure_clarity])
        tier, mode = self._categorize(aggregate)
        notes = {
            "avg_chars_per_page": round(avg_chars, 1),
            "figure_pressure": round(figure_pressure, 3),
        }
        return QualityProfile(
            ocr_confidence=text_density,
            image_quality=image_quality,
            structure_clarity=structure_clarity,
            aggregate=aggregate,
            tier=tier,
            processing_mode=mode,
            vision_signal=0.0,
            branch_consistency=1.0,
            branch_strategy="ocr-only",
            notes=notes,
        )

    # ---------------------------- post-ingestion --------------------------- #
    def summarize_chunks(
        self,
        chunks: Sequence[common.Chunk],
        page_images_dir: Path | None,
        preview: QualityProfile | None = None,
    ) -> QualityProfile:
        ocr_scores = []
        noise_scores = []
        consensus_scores = []
        branch_links = 0
        branch_mismatches = 0
        for chunk in chunks:
            metadata = chunk.metadata or {}
            backend = metadata.get("ocr_backend", "")
            consensus = metadata.get("ocr_consensus", 1.0)
            if backend == "pypdf":
                ocr_scores.append(1.0)
            elif backend == "none":
                ocr_scores.append(0.1)
            else:
                ocr_scores.append(0.6)
            consensus_scores.append(_clamp(float(consensus)))
            noise_scores.append(_clamp(1.0 - float(metadata.get("noise_score", 0.0))))
            branch_info = (metadata.get("branch_provenance") or {}).get("vision")
            region_type = (metadata.get("region_type") or "").lower()
            if isinstance(branch_info, dict):
                branch_links += 1
                branch_region = str(branch_info.get("region_type", "")).lower()
                if branch_region and region_type and branch_region != region_type:
                    branch_mismatches += 1
            elif region_type in {"figure", "equation", "table"}:
                branch_mismatches += 1
        ocr_confidence = mean(ocr_scores) if ocr_scores else (preview.ocr_confidence if preview else 0.5)
        consensus_avg = mean(consensus_scores) if consensus_scores else 1.0
        ocr_confidence = _clamp(0.7 * ocr_confidence + 0.3 * consensus_avg)
        structure_clarity = mean(noise_scores) if noise_scores else (preview.structure_clarity if preview else 0.5)
        image_quality = self._image_score(page_images_dir)
        vision_signal = _clamp(branch_links / max(1, len(chunks)))
        branch_consistency = 1.0 if branch_links == 0 else _clamp(1.0 - branch_mismatches / max(1, len(chunks)))
        branch_strategy = self._branch_strategy(vision_signal, branch_consistency)
        aggregate = mean([ocr_confidence, image_quality, structure_clarity, vision_signal])
        tier, mode = self._categorize(aggregate)
        notes = {
            "consensus": round(consensus_avg, 3),
            "chunks": len(chunks),
            "vision_branch_ratio": round(vision_signal, 3),
            "branch_consistency": round(branch_consistency, 3),
        }
        if preview:
            notes["preview_mode"] = preview.processing_mode
        return QualityProfile(
            ocr_confidence=ocr_confidence,
            image_quality=image_quality,
            structure_clarity=structure_clarity,
            aggregate=aggregate,
            tier=tier,
            processing_mode=mode,
            vision_signal=vision_signal,
            branch_consistency=branch_consistency,
            branch_strategy=branch_strategy,
            notes=notes,
        )

    # ------------------------------- helpers ------------------------------- #
    def _looks_structured(self, text: str) -> bool:
        if not text:
            return False
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return False
        headings = sum(1 for line in lines if line.isupper() or line.endswith(":"))
        enumerations = sum(1 for line in lines if line[:3].isdigit() or line.startswith(("-", "*")))
        return (headings + enumerations) >= max(1, len(lines) // 6)

    def _categorize(self, aggregate: float) -> tuple[str, str]:
        if aggregate <= self.poor_threshold:
            return "low", "aggressive"
        if aggregate >= self.excellent_threshold:
            return "high", "conservative"
        return "medium", "balanced"

    def _image_score(self, page_images_dir: Path | None, sample: int = 4) -> float:
        if not page_images_dir or not page_images_dir.exists():
            return 0.5
        sizes: List[float] = []
        for idx, image_path in enumerate(sorted(page_images_dir.glob("page_*.png"))):
            if idx >= sample:
                break
            try:
                with Image.open(image_path) as handle:
                    width, height = handle.size
            except Exception:
                continue
            megapixels = (width * height) / 1_000_000.0
            short_edge = min(width, height)
            normalized = 0.5 * _clamp(short_edge / 1200.0) + 0.5 * _clamp(megapixels / 2.0)
            sizes.append(normalized)
        if not sizes:
            return 0.5
        return _clamp(mean(sizes))

    def _branch_strategy(self, vision_signal: float, branch_consistency: float) -> str:
        if vision_signal < 0.2:
            return "ocr-only"
        if branch_consistency < 0.5:
            return "vision-scout"
        if vision_signal >= 0.6 and branch_consistency >= 0.6:
            return "vision-hybrid"
        return "vision-scout"


__all__ = ["InputQualityAssessor", "QualityProfile"]
