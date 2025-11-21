"""Branch B vision pipeline powered by Nougat + InternVL."""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from ..core import common
from ..models.model_adapters import (
    InternVLAdapter,
    InternVLConfig,
    get_shared_adapter,
    register_shared_adapter,
)
from ..models.nougat_adapter import NougatAdapter, NougatAdapterConfig
from ..models.math_ocr import MathSyntaxValidator, Pix2TexRecognizer
from .branch_orchestrator import BranchRunContext, BranchRunResult

LOGGER = logging.getLogger(__name__)


@dataclass
class BranchBConfig:
    models_dir: Path
    chunks_path: Path
    run_dir: Path
    artifacts_dir: Path
    max_regenerations: int = 2
    nougat_dir: Path | None = None
    pix2tex_dir: Path | None = None
    internvl_dir: Path | None = None
    preferred_device: str | None = None
    success_threshold: float = 0.65
    fallback_threshold: float = 0.4
    enable_layout_hints: bool = True

    @classmethod
    def from_context(cls, context: BranchRunContext) -> "BranchBConfig":
        shared = context.shared_context
        models_dir = Path(shared.get("models_dir", Path("latexify.models")))
        chunks_path = Path(shared.get("chunks_path"))
        run_dir = Path(shared.get("run_dir", context.branch_dir.parent))
        artifacts_dir = Path(shared.get("artifacts_dir", context.branch_dir.parent))
        max_regen = _safe_int(
            shared.get("branch_b_max_regenerations", os.environ.get("LATEXIFY_BRANCH_B_MAX_REGENERATIONS", 2)),
            default=2,
        )
        success_threshold = _safe_float(os.environ.get("LATEXIFY_BRANCH_B_SUCCESS_THRESHOLD"), default=0.65)
        fallback_threshold = _safe_float(os.environ.get("LATEXIFY_BRANCH_B_FALLBACK_THRESHOLD"), default=0.4)
        nougat_dir = _resolved_path(
            shared.get("branch_b_nougat_dir") or os.environ.get("LATEXIFY_BRANCH_B_NOUGAT_DIR"),
        )
        pix2tex_dir = _resolved_path(
            shared.get("branch_b_pix2tex_dir") or os.environ.get("LATEXIFY_BRANCH_B_PIX2TEX_DIR"),
        )
        internvl_dir = _resolved_path(
            shared.get("branch_b_internvl_dir")
            or os.environ.get("LATEXIFY_BRANCH_B_INTERNVL_DIR")
            or (models_dir / "ocr" / _sanitize_model_dir(os.environ.get("LATEXIFY_INTERNVL_MODEL", "OpenGVLab/InternVL3_5-8B"))),
        )
        if nougat_dir is None:
            candidates = [
                models_dir / "ocr" / "nougat-base",
                models_dir / "ocr" / "nougat-small",
            ]
            nougat_dir = next((candidate for candidate in candidates if candidate.exists()), None)
        if pix2tex_dir is None:
            candidate = models_dir / "ocr" / "pix2tex-base"
            pix2tex_dir = candidate if candidate.exists() else None
        return cls(
            models_dir=models_dir,
            chunks_path=chunks_path,
            run_dir=run_dir,
            artifacts_dir=artifacts_dir,
            max_regenerations=max_regen,
            nougat_dir=nougat_dir,
            pix2tex_dir=pix2tex_dir,
            internvl_dir=internvl_dir,
            preferred_device=context.resources.preferred_device,
            success_threshold=success_threshold,
            fallback_threshold=fallback_threshold,
            enable_layout_hints=os.environ.get("LATEXIFY_BRANCH_B_DISABLE_LAYOUT", "0") not in {"1", "true", "on"},
        )


@dataclass
class VisionBranchRegion:
    branch_id: str
    chunk_ids: List[str] = field(default_factory=list)
    region_type: str = "text"
    page: int = 0
    bbox: Tuple[float, float, float, float] | None = None
    extras: Dict[str, Any] = field(default_factory=dict)
    page_image: Path | None = None


class BranchBVisionPipeline:
    def __init__(self, config: BranchBConfig) -> None:
        self.config = config
        self.validator = MathSyntaxValidator()
        self._nougat: NougatAdapter | None = None
        self._pix2tex: Pix2TexRecognizer | None = None
        self._internvl: InternVLAdapter | None = None
        self._active_branch_dir: Path | None = None

    def run(self, context: BranchRunContext) -> BranchRunResult:
        context.ensure_directories()
        self._active_branch_dir = context.branch_dir
        if not self.config.chunks_path.exists():
            note = f"chunks missing: {self.config.chunks_path}"
            context.log("skipped", notes=note)
            self._active_branch_dir = None
            return BranchRunResult(branch=context.name, status="skipped", notes=note)
        regions = self._collect_regions()
        if not regions:
            context.log("skipped", notes="no vision branches")
            self._active_branch_dir = None
            return BranchRunResult(branch=context.name, status="skipped", notes="no vision branches")
        context.log("running", notes=f"regions={len(regions)}")
        processed: List[Dict[str, Any]] = []
        for region in regions:
            context.record_progress("processing", branch_id=region.branch_id)
            result = self._process_region(region)
            processed.append(result)
            context.record_progress(
                "processed",
                branch_id=region.branch_id,
                confidence=result.get("confidence", 0.0),
                backend=result.get("backend"),
            )
        report_path = context.report_dir / "branch_b_vision.json"
        payload = {
            "branch": context.name,
            "results": processed,
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        success_conf = [entry["confidence"] for entry in processed if entry.get("status") == "ok"]
        avg_conf = round(mean(success_conf), 3) if success_conf else 0.0
        max_conf = round(max(success_conf), 3) if success_conf else 0.0
        failures = len([entry for entry in processed if entry.get("status") != "ok"])
        metrics = {
            "outputs": float(len(processed)),
            "avg_confidence": avg_conf,
            "max_confidence": max_conf,
            "failures": float(failures),
        }
        status = "completed" if success_conf else "failed"
        context.log(status, output_files=[str(report_path)], metrics=metrics)
        metadata = {
            "regions": len(processed),
            "success": len(success_conf),
            "models": ["nougat", "pix2tex", "internvl"],
        }
        self._active_branch_dir = None
        return BranchRunResult(
            branch=context.name,
            status=status,
            output_files=[report_path, context.branch_dir],
            metrics=metrics,
            metadata=metadata,
            notes="vision branch synthesis",
        )

    # Internal helpers -----------------------------------------------------

    def _collect_regions(self) -> List[VisionBranchRegion]:
        chunks = common.load_chunks(self.config.chunks_path)
        artifacts_dir = self.config.artifacts_dir
        run_dir = self.config.run_dir
        regions: Dict[str, VisionBranchRegion] = {}
        for chunk in chunks:
            metadata = chunk.metadata or {}
            provenance = metadata.get("branch_provenance") or {}
            vision_meta = provenance.get("vision")
            if not isinstance(vision_meta, dict):
                continue
            branch_id = str(vision_meta.get("branch_id") or "").strip()
            if not branch_id:
                continue
            entry = regions.setdefault(
                branch_id,
                VisionBranchRegion(
                    branch_id=branch_id,
                    region_type=str(vision_meta.get("region_type", metadata.get("region_type", "text"))),
                    page=int(vision_meta.get("page", chunk.page or 0)),
                    bbox=_coerce_bbox(vision_meta.get("bbox")),
                    extras=dict(vision_meta.get("extras") or {}),
                    page_image=_locate_image(vision_meta.get("page_image"), artifacts_dir, run_dir),
                ),
            )
            entry.chunk_ids.append(chunk.chunk_id)
            if not entry.page_image:
                entry.page_image = _locate_image(vision_meta.get("page_image"), artifacts_dir, run_dir)
        return list(regions.values())

    def _process_region(self, region: VisionBranchRegion) -> Dict[str, Any]:
        record = {
            "branch_id": region.branch_id,
            "chunk_ids": region.chunk_ids,
            "region_type": region.region_type,
            "page": region.page,
            "bbox": region.bbox,
        }
        if not region.page_image or not region.page_image.exists():
            record.update({"status": "no_image", "confidence": 0.0})
            return record
        layout_hint = self._describe_layout(region.page_image) if self.config.enable_layout_hints else None
        latex, backend, attempts, confidence, errors = self._transcribe(region, layout_hint)
        record.update(
            {
                "backend": backend,
                "attempts": attempts,
                "confidence": confidence,
                "layout_hint": layout_hint,
                "errors": errors,
            }
        )
        if not latex:
            record.update({"status": "empty"})
            return record
        try:
            region_dir = self._region_dir(region.branch_id)
            region_dir.mkdir(parents=True, exist_ok=True)
            latex_path = region_dir / "main.tex"
            latex_path.write_text(latex, encoding="utf-8")
            meta_path = region_dir / "metadata.json"
            metadata = {
                "branch_id": region.branch_id,
                "backend": backend,
                "confidence": confidence,
                "layout_hint": layout_hint,
                "attempts": attempts,
            }
            meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
            record.update(
                {
                    "status": "ok",
                    "latex_path": str(latex_path),
                    "metadata_path": str(meta_path),
                }
            )
        except Exception as exc:
            LOGGER.warning("Failed to write branch artifacts for %s: %s", region.branch_id, exc)
            record.update({"status": "io_error", "confidence": 0.0})
        return record

    def _transcribe(
        self,
        region: VisionBranchRegion,
        layout_hint: Optional[str],
    ) -> Tuple[str, str, int, float, List[str]]:
        errors: List[str] = []
        attempts = 0
        best_conf = 0.0
        best_text = ""
        backend = "nougat"
        nougat = self._load_nougat()
        target_image = region.page_image
        if nougat and target_image:
            for _ in range(self.config.max_regenerations + 1):
                attempts += 1
                try:
                    candidate, raw_conf = nougat.predict_with_confidence(target_image)
                except Exception as exc:  # pragma: no cover - heavy dependency
                    errors.append(str(exc))
                    continue
                sanitized = self._aestheticize(candidate, region.region_type)
                layout_score = self._score_candidate(sanitized, layout_hint, region.region_type)
                confidence = round(0.6 * raw_conf + 0.4 * layout_score, 3)
                if confidence > best_conf:
                    best_conf = confidence
                    best_text = sanitized
                if confidence >= self.config.success_threshold:
                    break
        if (not best_text or best_conf < self.config.fallback_threshold) and target_image:
            fallback = self._run_pix2tex(target_image)
            if fallback:
                backend = "pix2tex"
                attempts += 1
                sanitized = self._aestheticize(fallback, region.region_type)
                confidence = self._score_candidate(sanitized, layout_hint, region.region_type)
                if confidence > best_conf:
                    best_conf = confidence
                    best_text = sanitized
        return best_text, backend, attempts, round(best_conf, 3), errors

    def _run_pix2tex(self, image_path: Path) -> str | None:
        recognizer = self._load_pix2tex()
        if recognizer is None:
            return None
        try:
            image = Image.open(image_path).convert("L")
        except Exception as exc:  # pragma: no cover - io/issues
            LOGGER.debug("pix2tex image load failed: %s", exc)
            return None
        try:
            return recognizer.predict(image)
        except Exception as exc:  # pragma: no cover - optional dep
            LOGGER.debug("pix2tex inference failed: %s", exc)
            return None

    def _load_pix2tex(self) -> Pix2TexRecognizer | None:
        if self._pix2tex is None:
            self._pix2tex = Pix2TexRecognizer(self.config.pix2tex_dir)
        return self._pix2tex

    def _load_nougat(self) -> NougatAdapter | None:
        if self._nougat is not None or self.config.nougat_dir is None:
            return self._nougat
        try:
            self._nougat = NougatAdapter(
                NougatAdapterConfig(
                    model_dir=self.config.nougat_dir,
                    max_new_tokens=1024,
                    fallback_pix2tex_dir=self.config.pix2tex_dir,
                ),
            )
        except Exception as exc:  # pragma: no cover - heavy dep
            LOGGER.warning("Nougat unavailable for Branch B: %s", exc)
            self._nougat = None
        return self._nougat

    def _load_internvl(self) -> InternVLAdapter | None:
        if self._internvl is not None:
            return self._internvl
        shared = None
        try:
            candidate = get_shared_adapter("internvl")
            if isinstance(candidate, InternVLAdapter):
                shared = candidate
        except Exception:
            shared = None
        if shared is not None:
            self._internvl = shared
            return self._internvl
        if not self.config.internvl_dir or not self.config.internvl_dir.exists():
            return None
        try:
            adapter = InternVLAdapter(
                InternVLConfig(
                    model_dir=self.config.internvl_dir,
                    device=self.config.preferred_device,
                    prompt=_layout_prompt(),
                )
            )
            register_shared_adapter("internvl", adapter)
            self._internvl = adapter
        except Exception as exc:  # pragma: no cover - heavy dep
            LOGGER.warning("InternVL unavailable for Branch B: %s", exc)
            self._internvl = None
        return self._internvl

    def _describe_layout(self, image_path: Path) -> str | None:
        adapter = self._load_internvl()
        if adapter is None:
            return None
        original_prompt = getattr(adapter, "prompt", None)
        new_prompt = _layout_prompt()
        try:
            adapter.prompt = new_prompt
            return adapter.predict(image_path)
        except Exception as exc:  # pragma: no cover - best effort
            LOGGER.debug("InternVL layout hint failed: %s", exc)
            return None
        finally:
            if original_prompt is not None:
                adapter.prompt = original_prompt

    def _score_candidate(self, latex: str, layout_hint: Optional[str], region_type: str) -> float:
        if not latex.strip():
            return 0.0
        syntax_score = self.validator.score(latex)
        length_score = min(1.0, len(latex) / 1200)
        layout_score = self._layout_similarity(latex, layout_hint)
        region_bonus = 0.1 if region_type in {"figure", "table", "diagram"} else 0.0
        score = 0.45 * syntax_score + 0.35 * layout_score + 0.2 * length_score + region_bonus
        return max(0.0, min(1.0, round(score, 3)))

    def _layout_similarity(self, latex: str, layout_hint: Optional[str]) -> float:
        if not layout_hint:
            return 0.5
        hint_tokens = _tokenize(layout_hint)
        latex_tokens = _tokenize(latex)
        if not hint_tokens or not latex_tokens:
            return 0.4
        overlap = len(hint_tokens & latex_tokens) / max(1, len(hint_tokens))
        return round(max(0.2, min(1.0, overlap + 0.2)), 3)

    def _aestheticize(self, latex: str, region_type: str) -> str:
        text = latex.strip()
        if not text:
            return text
        if region_type == "figure" and "\\begin{figure" not in text:
            return "\\begin{figure}[ht]\n\\centering\n" + text + "\n\\end{figure}"
        if region_type == "table" and "\\begin{table" not in text:
            return "\\begin{table}[ht]\n\\centering\n" + text + "\n\\end{table}"
        if region_type in {"display-equation", "equation"} and not text.startswith("\\["):
            return "\\[\n" + text + "\n\\]"
        return text

    def _region_dir(self, branch_id: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9._-]+", "-", branch_id).strip("-") or branch_id
        base = self._active_branch_dir or (self.config.artifacts_dir / "branches" / "vision_b")
        return base / safe


def run_branch(context: BranchRunContext) -> BranchRunResult:
    config = BranchBConfig.from_context(context)
    pipeline = BranchBVisionPipeline(config)
    return pipeline.run(context)


# Utility helpers --------------------------------------------------------------

def _coerce_bbox(values: Any) -> Tuple[float, float, float, float] | None:
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        return None
    try:
        return tuple(float(v) for v in values)  # type: ignore[return-value]
    except (TypeError, ValueError):
        return None


def _locate_image(value: Any, artifacts_dir: Path, run_dir: Path) -> Path | None:
    if not value:
        return None
    candidates = []
    path = Path(str(value))
    candidates.append(path)
    if not path.is_absolute():
        candidates.append(artifacts_dir / path)
        candidates.append(run_dir / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _sanitize_model_dir(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).lower()


def _resolved_path(raw: Any) -> Path | None:
    if not raw:
        return None
    path = Path(str(raw)).expanduser()
    return path if path.exists() else None


def _tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[A-Za-z0-9]+", text.lower())
    return set(tokens)


def _layout_prompt() -> str:
    return (
        "Describe the layout of this scientific region. Mention if it is a figure, table, or equation "
        "and summarize key captions or variables. Keep it short."
    )


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


__all__ = ["run_branch", "BranchBVisionPipeline", "BranchBConfig"]
