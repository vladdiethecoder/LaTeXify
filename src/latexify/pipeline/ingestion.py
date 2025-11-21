"""Ingestion stage for the self-contained LaTeXify latexify.pipeline."""
from __future__ import annotations

import importlib
import json
import logging
import math
import os
import re
import subprocess
import contextlib
import time
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple, Set

try:  # pragma: no cover - optional dependency
    from huggingface_hub import snapshot_download
except Exception:  # pragma: no cover
    snapshot_download = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from pdf2image import convert_from_path
except Exception:  # pragma: no cover
    convert_from_path = None

try:  # pragma: no cover - optional dependency
    import fitz  # type: ignore
except Exception:  # pragma: no cover
    fitz = None  # type: ignore

from PIL import Image

try:  # pragma: no cover - optional dependency
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import torch
    import torch.nn.functional as F  # type: ignore
except Exception:  # pragma: no cover
    torch = None
    F = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor  # type: ignore
except Exception:  # pragma: no cover
    LayoutLMv3ForTokenClassification = None  # type: ignore
    LayoutLMv3Processor = None  # type: ignore

from pypdf import PdfReader

try:  # pragma: no cover - OCR dependency is optional on some hosts
    import pytesseract
except Exception:  # pragma: no cover
    pytesseract = None

try:  # pragma: no cover - optional dependency
    from ..models.math_ocr import MathOCREngine
except Exception:  # pragma: no cover
    MathOCREngine = None  # type: ignore

try:  # pragma: no cover - shared adapter helpers
    from ..models.model_adapters import release_shared_adapter as _release_shared_adapter
except Exception:  # pragma: no cover
    _release_shared_adapter = None  # type: ignore


from ..core import common
from ..core.model_paths import resolve_models_root
from ..core.config import BackendToggleConfig
from ..utils.ensemble import EnsembleVoter
from ..models.monkey_ocr_adapter import MonkeyOCRAdapter, MonkeyOCRPageResult
from ..models.vlm_adapters import get_vlm_adapter
from .semantic_chunking import SemanticChunker
from .sectioning import LLMSectioner, build_sectioner
from .ambiguity_resolver import AmbiguityResolver
from .math_support import MathContentClassifier
from .surya_adapter import SuryaLayoutDetector
from .quality_assessor import InputQualityAssessor, QualityProfile

LOGGER = logging.getLogger(__name__)
DEFAULT_CHUNK_CHARS = 1200
MODELS_ROOT = resolve_models_root(Path(__file__).resolve().parents[1] / "models")
SURYA_MODEL_SUBDIR = Path("layout") / "surya"
OCR_MODES = {
    "auto",
    "pytesseract",
    "nougat",
    "florence2",
    "internvl",
    "qwenvl",
    "mathvision",
    "mathocr",
    "monkeyocr",
    "none",
}


def _layout_conf_threshold() -> float:
    try:
        return float(os.environ.get("LATEXIFY_LAYOUT_CONFIDENCE_THRESHOLD", "0.0"))
    except ValueError:
        return 0.0


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "off", ""}
def _sanitize_model_subdir(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).lower()


def _bbox_to_polygon(bbox: Tuple[float, float, float, float]) -> List[Tuple[float, float]]:
    x0, y0, x1, y1 = bbox
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


INTERNVL_MODEL_ID = os.environ.get("LATEXIFY_INTERNVL_MODEL", "OpenGVLab/InternVL3_5-8B")
INTERNVL_MODEL_SUBPATH = Path("ocr") / _sanitize_model_subdir(INTERNVL_MODEL_ID)
QWEN_VL_MODEL_ID = os.environ.get("LATEXIFY_QWEN_VL_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
QWEN_VL_MODEL_SUBPATH = Path("ocr") / _sanitize_model_subdir(QWEN_VL_MODEL_ID)
QWEN_VL_PROMPT = os.environ.get(
    "LATEXIFY_QWEN_VL_PROMPT",
    "Transcribe this entire PDF page into high-fidelity LaTeX, preserving math, tables, and structure.",
)
QWEN_VL_MAX_NEW_TOKENS = int(os.environ.get("LATEXIFY_QWEN_VL_MAX_NEW_TOKENS", "768"))
QWEN_VL_TEMPERATURE = float(os.environ.get("LATEXIFY_QWEN_VL_TEMPERATURE", "0.0"))
QWEN_VL_TOP_P = float(os.environ.get("LATEXIFY_QWEN_VL_TOP_P", "0.9"))
QWEN_VL_DEVICE_ENV = os.environ.get("LATEXIFY_QWEN_VL_DEVICE", "auto").strip().lower()
QWEN_VL_LOAD_IN_8BIT = _env_flag("LATEXIFY_QWEN_VL_LOAD_IN_8BIT")
QWEN_VL_LOAD_IN_4BIT = _env_flag("LATEXIFY_QWEN_VL_LOAD_IN_4BIT")
QWEN_VL_MAX_GPU_RETRIES = int(os.environ.get("LATEXIFY_QWEN_VL_MAX_GPU_RETRIES", "2"))

OCR_MODEL_SPECS = {
    "nougat": {"repo_id": "facebook/nougat-small", "subpath": Path("ocr") / "nougat-small"},
    "florence2": {"repo_id": "microsoft/Florence-2-large-ft", "subpath": Path("ocr") / "florence-2-large"},
    "internvl": {"repo_id": INTERNVL_MODEL_ID, "subpath": INTERNVL_MODEL_SUBPATH},
    "qwenvl": {"repo_id": QWEN_VL_MODEL_ID, "subpath": QWEN_VL_MODEL_SUBPATH},
    "mathvision": {"repo_id": "microsoft/trocr-base-handwritten", "subpath": Path("ocr") / "trocr-math"},
    "mathocr": {"repo_id": "lupantech/pix2tex-base", "subpath": Path("ocr") / "pix2tex-base"},
}
SYSTEM_MEMORY_SKIP_HEAVY_GB = float(os.environ.get("LATEXIFY_SYSTEM_MEMORY_SKIP_HEAVY_GB", "4"))
FORCE_HEAVY_OCR = os.environ.get("LATEXIFY_OCR_FORCE_HEAVY", "0") == "1"
GPU_PREF_ENV = os.environ.get("LATEXIFY_OCR_GPU_PREF")
LAYOUTLM_MODEL_OVERRIDE = os.environ.get("LATEXIFY_LAYOUTLM_MODEL")
LAYOUTLM_DEVICE_ENV = os.environ.get("LATEXIFY_LAYOUTLM_DEVICE", "auto")
CLIP_DEVICE_ENV = os.environ.get("LATEXIFY_CLIP_DEVICE", "auto")
SEQUENTIAL_OCR = os.environ.get("LATEXIFY_OCR_SEQUENTIAL", "1") != "0"
MATHOCR_SEMANTIC_VALIDATION = os.environ.get("LATEXIFY_MATHOCR_SEMANTIC_VALIDATION", "0") == "1"


def _parse_force_gpu_device(value: str | None) -> str | None:
    if not value:
        return None
    normalized = value.strip().lower()
    if normalized in {"", "0", "false", "off"}:
        return None
    if normalized in {"1", "true", "on"}:
        return "cuda:0"
    if normalized.isdigit():
        return f"cuda:{normalized}"
    if normalized.startswith("cuda:"):
        return normalized
    return normalized


def _parse_force_gpu_backends(value: str | None) -> Set[str]:
    if not value:
        return set()
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "all"}:
        return {"all"}
    tokens: Set[str] = set()
    for token in re.split(r"[,\s]+", value):
        token = token.strip()
        if token:
            tokens.add(token.lower())
    return tokens


def _parse_vram_headroom_bytes() -> int:
    raw = os.environ.get("LATEXIFY_OCR_VRAM_HEADROOM_GB")
    try:
        value = float(raw) if raw is not None else None
    except (TypeError, ValueError):
        LOGGER.warning(
            "Invalid LATEXIFY_OCR_VRAM_HEADROOM_GB value %s; using automatic safety margin.",
            raw,
        )
        value = None
    if value is None:
        total_gb = None
        if torch is not None and torch.cuda.is_available():
            with contextlib.suppress(Exception):
                total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        value = 1.0 if total_gb and total_gb <= 32 else 2.5
    value = max(0.0, value)
    return int(value * 1024**3)


OCR_VRAM_HEADROOM_BYTES = _parse_vram_headroom_bytes()
FORCE_GPU_OCR_DEVICE = _parse_force_gpu_device(os.environ.get("LATEXIFY_FORCE_GPU_OCR"))
FORCE_GPU_OCR_BACKENDS = {"florence2", "internvl", "qwenvl", "mathvision", "mathocr"}
FORCE_GPU_BACKEND_OVERRIDES = _parse_force_gpu_backends(os.environ.get("LATEXIFY_OCR_FORCE_GPU"))


def _resolve_qwenvl_device() -> str:
    pref = QWEN_VL_DEVICE_ENV
    if pref == "auto" or not pref:
        return "cuda:0" if not LIMITED_GPU else "cpu"
    if pref == "cuda":
        return "cuda:0"
    return pref


def _detect_limited_gpu() -> bool:
    if torch is None or not torch.cuda.is_available():
        return True
    try:
        device_count = torch.cuda.device_count()
    except Exception:
        device_count = 0
    if device_count >= 2:
        return False
    total_bytes = 0
    try:
        total_bytes = torch.cuda.get_device_properties(0).total_memory
    except Exception:
        total_bytes = 0
    max_supported = 40 * 1024**3
    if total_bytes and total_bytes > max_supported:
        return False
    return True


LIMITED_GPU = _detect_limited_gpu()

if LIMITED_GPU:
    if LAYOUTLM_DEVICE_ENV == "auto":
        LAYOUTLM_DEVICE_ENV = "cpu"
    if CLIP_DEVICE_ENV == "auto":
        CLIP_DEVICE_ENV = "cpu"

PREFER_QWEN_VL = _env_flag("LATEXIFY_PREFER_QWEN_VL", default=LIMITED_GPU)


def _resolve_release_mode(value: str | None, limited: bool) -> str:
    """Determine how aggressively we should unload heavy OCR backends."""

    if value:
        normalized = value.strip().lower()
        if normalized in {"run", "page"}:
            return normalized
        LOGGER.warning(
            "Ignoring invalid LATEXIFY_OCR_RELEASE_MODE value '%s'; falling back to auto-detect.",
            value,
        )
    return "page" if limited else "run"


RELEASE_HEAVY_MODE = _resolve_release_mode(os.environ.get("LATEXIFY_OCR_RELEASE_MODE"), LIMITED_GPU)
BACKEND_PYTHON_DEPS = {
    "florence2": ("einops", "timm"),
    "internvl": ("einops",),
    "qwenvl": ("torch", "transformers"),
    "mathocr": ("pix2tex", "ultralytics"),
}
FORMULA_RE = re.compile(r"(\\begin\{equation\}|\\frac|\\sum|\\int|=|\\[a-z]+)")
TABLE_BORDER_RE = re.compile(r"(\+[-+]+\+)|(\|.+\|)")
LIST_BULLET_RE = re.compile(r"^([0-9]+\.[\)\s]|[A-Za-z]\.|[-*•])\s+")
HEADER_KEYWORDS = ("chapter", "section", "appendix", "part", "lesson")
WORD_RE = re.compile(r"[A-Za-z]+")
QUESTION_RE = re.compile(r"^(question|q)\s*([0-9]+[a-z]?|\([^)]+\))", re.IGNORECASE)
ANSWER_RE = re.compile(r"^(answer|solution)\b", re.IGNORECASE)
DIGIT_GAP_RE = re.compile(r"(?<=\d)\s+(?=\d)")


@dataclass
class IngestionResult:
    chunks_path: Path
    image_dir: Path
    ocr_dir: Path
    document_path: Path | None = None
    page_images_dir: Path | None = None
    page_images_available: bool = False
    tree_path: Path | None = None
    manifest_path: Path | None = None
    quality_profile: Dict[str, object] | None = None
    vision_branch_summary: Dict[str, object] | None = None
    master_ocr_items_path: Path | None = None


@dataclass
class MasterOCRItem:
    item_id: str
    page: int
    region_type: str
    polygon: List[Tuple[float, float]]
    bbox: Tuple[float, float, float, float]
    content: str | None = None
    page_width: float | None = None
    page_height: float | None = None

    def to_payload(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "id": self.item_id,
            "page": self.page,
            "region_type": self.region_type,
            "polygon": [[float(x), float(y)] for x, y in self.polygon],
            "bbox": [float(v) for v in self.bbox],
        }
        if self.content:
            payload["content"] = self.content
        if self.page_width:
            payload["page_width_pt"] = float(self.page_width)
        if self.page_height:
            payload["page_height_pt"] = float(self.page_height)
        return payload


def _export_master_ocr_items(layout_segments: Dict[int, List[LayoutRegion]], target: Path) -> Path | None:
    if not layout_segments:
        return None
    items: List[Dict[str, object]] = []
    for page_idx, regions in layout_segments.items():
        for order, region in enumerate(regions):
            polygon = region.extras.get("polygon") if region.extras else None
            if not polygon:
                polygon = _bbox_to_polygon(region.bbox)
            extras = region.extras or {}
            page_width = extras.get("page_width_pt") or extras.get("page_width")
            page_height = extras.get("page_height_pt") or extras.get("page_height")
            payload = MasterOCRItem(
                item_id=f"page{page_idx + 1:04d}_region{order:03d}",
                page=page_idx + 1,
                region_type=region.tag,
                polygon=[tuple(point) for point in polygon],
                bbox=region.bbox,
                content=(region.text or extras.get("text")),
                page_width=float(page_width) if page_width else None,
                page_height=float(page_height) if page_height else None,
            ).to_payload()
            items.append(payload)
    if not items:
        return None
    target.write_text(json.dumps(items, indent=2), encoding="utf-8")
    return target


class PageImageStore:
    """Lazily rasterizes PDF pages to PNG files for downstream models."""

    def __init__(self, pdf_path: Path, cache_dir: Path, enabled: bool) -> None:
        self.pdf_path = pdf_path
        self.cache_dir = cache_dir
        self._pymupdf_doc = None
        self.enabled = enabled and (fitz is not None or convert_from_path is not None)
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _render_with_pymupdf(self, page_index: int, target: Path) -> bool:
        if fitz is None:
            return False
        try:
            if self._pymupdf_doc is None:
                self._pymupdf_doc = fitz.open(str(self.pdf_path))
            page = self._pymupdf_doc.load_page(page_index)
            pix = page.get_pixmap(dpi=300)
            pix.save(str(target))
            return target.exists()
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("PyMuPDF rasterization failed for page %s: %s", page_index + 1, exc)
            return False

    def get_page_image(self, page_index: int) -> Path | None:
        if not self.enabled:
            return None
        candidate = self.cache_dir / f"page_{page_index + 1:04d}.png"
        if candidate.exists():
            return candidate
        if self._render_with_pymupdf(page_index, candidate):
            return candidate
        if convert_from_path is None:
            LOGGER.warning(
                "Unable to rasterize page %s: pdf2image unavailable and PyMuPDF failed.",
                page_index + 1,
            )
            self.enabled = False
            return None
        try:
            convert_from_path(
                str(self.pdf_path),
                first_page=page_index + 1,
                last_page=page_index + 1,
                fmt="png",
                output_folder=str(self.cache_dir),
                output_file=f"page_{page_index + 1:04d}",
            )
        except Exception as exc:  # pragma: no cover - environment specific
            LOGGER.warning("Failed to rasterize page %s: %s", page_index + 1, exc)
            self.enabled = False
            return None
        return candidate if candidate.exists() else None


@dataclass
class LayoutRegion:
    """Structured description of a page fragment returned by LayoutAnalyzer."""

    text: str
    tag: str
    bbox: Tuple[float, float, float, float]
    column: int
    order: int
    font_size: float
    extras: Dict[str, object]


class LayoutAnalyzer:
    """Lightweight layout analyzer powered by PyMuPDF blocks."""

    def __init__(self, pdf_path: Path, enabled: bool = True) -> None:
        self.pdf_path = pdf_path
        self.enabled = enabled and fitz is not None
        self._doc = None
        self._page_width: float | None = None

    def _load_doc(self):
        if self._doc is None and self.enabled:
            try:
                self._doc = fitz.open(str(self.pdf_path))
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("LayoutAnalyzer failed to open %s: %s", self.pdf_path, exc)
                self.enabled = False
        return self._doc

    def close(self) -> None:
        if self._doc is not None:
            try:
                self._doc.close()
            except Exception:
                pass
            self._doc = None

    def analyze_document(self, page_count: int) -> Dict[int, List[LayoutRegion]]:
        if not self.enabled or not self._load_doc():
            return {}
        regions: Dict[int, List[LayoutRegion]] = {}
        for page_idx in range(page_count):
            try:
                regions[page_idx] = self._analyze_page(page_idx)
            except Exception as exc:  # pragma: no cover - best effort guard
                LOGGER.debug("LayoutAnalyzer skipped page %s: %s", page_idx + 1, exc)
                continue
        return regions

    def _analyze_page(self, page_index: int) -> List[LayoutRegion]:
        doc = self._load_doc()
        if doc is None:
            return []
        page = doc.load_page(page_index)
        width = page.rect.width or 1.0
        self._page_width = width
        blocks = page.get_text("dict").get("blocks", [])
        regions: List[LayoutRegion] = []
        order = 0
        for block in blocks:
            if block.get("type") != 0:
                continue
            lines = block.get("lines", [])
            if not lines:
                continue
            spans = [span for line in lines for span in line.get("spans", []) if span.get("text")]
            if not spans:
                continue
            text = " ".join(span.get("text", "").strip() for span in spans).strip()
            if not text:
                continue
            font_size = max(span.get("size", 0.0) or 0.0 for span in spans)
            bbox = tuple(block.get("bbox", (0, 0, 0, 0)))
            column = self._infer_column(bbox, width)
            tag, extras = self._classify_text(text, font_size)
            extras.update({
                "column": column,
                "bbox": bbox,
                "font_size": font_size,
                "page_index": page_index + 1,
                "page_width_pt": width,
                "page_height_pt": page.rect.height or 1.0,
                "layout_confidence": self._confidence_from_font(font_size),
            })
            extras["branch_id"] = f"vision_page{page_index + 1:03d}_region{order:03d}"
            regions.append(LayoutRegion(text=text, tag=tag, bbox=bbox, column=column, order=order, font_size=font_size, extras=extras))
            order += 1
        return regions

    def _infer_column(self, bbox: Tuple[float, float, float, float], page_width: float) -> int:
        center = (bbox[0] + bbox[2]) / 2 if bbox else 0
        normalized = center / page_width if page_width else 0
        if normalized < 0.4:
            return 1
        if normalized > 0.65:
            return 2
        return 1

    @staticmethod
    def _confidence_from_font(font_size: float) -> float:
        normalized = min(1.0, max(0.0, font_size / 24.0))
        return round(0.4 + 0.6 * normalized, 3)

    def _classify_text(self, text: str, font_size: float) -> Tuple[str, Dict[str, object]]:
        lowered = text.lower().strip()
        extras: Dict[str, object] = {}
        if QUESTION_RE.match(lowered):
            extras["question_label"] = QUESTION_RE.match(lowered).group(0)
            return "question", extras
        if ANSWER_RE.match(lowered):
            return "answer", extras
        if lowered.startswith("figure") or lowered.startswith("fig."):
            return "figure", extras
        if lowered.startswith("table") or any(TABLE_BORDER_RE.search(line) for line in text.splitlines()):
            return "table", extras
        if LIST_BULLET_RE.match(text.strip()):
            extras["list_depth"] = estimate_list_depth(text.splitlines())
            return "list", extras
        if font_size >= 18 or detect_header_level(text) > 0:
            extras["header_level"] = detect_header_level(text)
            return "heading", extras
        if FORMULA_RE.search(text):
            extras["formula_detected"] = True
            return "formula", extras
        return "text", extras


DEFAULT_LAYOUTLM_MODEL = "microsoft/layoutlmv3-base"


@dataclass
class VisionSynthesisBranch:
    """Branch metadata describing regions targeted for vision synthesis."""

    branch_id: str
    page_index: int
    region_type: str
    bbox: Tuple[float, float, float, float] | None
    extras: Dict[str, object] = field(default_factory=dict)
    page_image: str | None = None

    def to_metadata(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "branch_id": self.branch_id,
            "page": self.page_index,
            "region_type": self.region_type,
        }
        if self.bbox:
            payload["bbox"] = [float(value) for value in self.bbox]
        if self.page_image:
            payload["page_image"] = self.page_image
        if self.extras:
            payload["extras"] = self.extras
        return payload


class DocumentStructureAnalyzer:
    """Higher-level structural analyzer that emits per-page document trees."""

    def __init__(
        self,
        pdf_path: Path,
        enable_layoutlm: bool = True,
        model_name: str | None = None,
        label_map: Dict[int, str] | None = None,
        enable_vision_branch: bool = False,
        monkey_adapter: MonkeyOCRAdapter | None = None,
        layout_backend: str | None = None,
        page_image_store: PageImageStore | None = None,
    ) -> None:
        backend_env = (os.environ.get("LATEXIFY_LAYOUT_BACKEND", "pymupdf") or "pymupdf").lower()
        if layout_backend and layout_backend.strip():
            backend_env = layout_backend.strip().lower()
        self._layout_backend = backend_env
        self._surya_math_detector = os.environ.get("LATEXIFY_SURYA_MATH_DETECTOR", "1") not in {"0", "false"}
        self._surya_detector: SuryaLayoutDetector | None = None
        self._page_image_store = page_image_store
        if self._layout_backend == "surya" and self._page_image_store is None:
            cache_dir = pdf_path.parent / ".surya_cache"
            self._page_image_store = PageImageStore(pdf_path, cache_dir, enabled=True)
        self._pdf_path = pdf_path
        self._page_sizes = self._resolve_page_sizes(pdf_path)
        self._layout = LayoutAnalyzer(pdf_path, enabled=self._layout_backend != "surya")
        self._label_map = label_map or {}
        self._tree_cache: Dict[int, List[Dict[str, object]]] | None = None
        self._use_layoutlm = (
            enable_layoutlm
            and LayoutLMv3ForTokenClassification is not None
            and LayoutLMv3Processor is not None
            and torch is not None
        )
        self._model = None
        self._processor = None
        self._device = None
        self._vision_branch_enabled = enable_vision_branch
        self._vision_branch_cache: Dict[str, VisionSynthesisBranch] = {}
        self._monkey_adapter = monkey_adapter
        resolved_model_name = model_name or LAYOUTLM_MODEL_OVERRIDE or DEFAULT_LAYOUTLM_MODEL
        self._resolved_model_name = resolved_model_name
        if self._use_layoutlm and self._layout_backend != "surya":
            try:  # pragma: no cover - heavy dependency
                device = self._resolve_device()
                LOGGER.info("Loading LayoutLMv3 weights from %s on %s", resolved_model_name, device)
                dtype = torch.float16 if device.type == "cuda" else torch.float32
                processor_source = resolved_model_name
                try:
                    self._processor = LayoutLMv3Processor.from_pretrained(processor_source)
                except Exception as proc_exc:
                    if processor_source != DEFAULT_LAYOUTLM_MODEL:
                        LOGGER.warning(
                            "LayoutLM processor unavailable at %s (%s); falling back to %s.",
                            processor_source,
                            proc_exc,
                            DEFAULT_LAYOUTLM_MODEL,
                        )
                        processor_source = DEFAULT_LAYOUTLM_MODEL
                        self._processor = LayoutLMv3Processor.from_pretrained(processor_source)
                    else:
                        raise
                self._model = LayoutLMv3ForTokenClassification.from_pretrained(
                    resolved_model_name,
                    torch_dtype=dtype,
                )
                self._model.to(device)
                self._model.eval()
                self._device = device
            except Exception as exc:
                LOGGER.warning("LayoutLMv3 unavailable (%s); falling back to heuristics for document structure.", exc)
                self._model = None
                self._processor = None
                self._use_layoutlm = False

    def analyze_document(
        self, page_count: int
    ) -> Tuple[Dict[int, List[LayoutRegion]], Dict[int, List[Dict[str, object]]]]:
        if self._layout_backend == "surya":
            layout_regions = self._surya_layout(page_count)
        elif self._monkey_adapter and self._monkey_adapter.available():
            layout_regions = self._monkey_layout(page_count)
        else:
            layout_regions = self._layout.analyze_document(page_count)
        tree: Dict[int, List[Dict[str, object]]] = {}
        for page_idx, regions in layout_regions.items():
            nodes: List[Dict[str, object]] = []
            for idx, region in enumerate(regions):
                label = self._predict_label(region)
                nodes.append(
                    {
                        "node_id": f"page{page_idx + 1:04d}_node{idx:03d}",
                        "label": label,
                        "bbox": region.bbox,
                        "column": region.column,
                        "order": region.order,
                        "text": region.text,
                        "metadata": region.extras,
                    }
                )
            tree[page_idx] = nodes
        self._tree_cache = tree
        return layout_regions, tree

    def _monkey_layout(self, page_count: int) -> Dict[int, List[LayoutRegion]]:
        assert self._monkey_adapter is not None
        regions_map: Dict[int, List[LayoutRegion]] = {}
        raw_regions = self._monkey_adapter.layout_regions_for_document(page_count)
        for page_idx, regions in raw_regions.items():
            converted: List[LayoutRegion] = []
            for region in regions:
                extras = dict(region.metadata)
                extras.setdefault("layout_confidence", region.confidence)
                extras.setdefault("monkey_backend", "monkeyocr")
                converted.append(
                    LayoutRegion(
                        text=region.text,
                        tag=region.region_type,
                        bbox=region.bbox,
                        column=region.column,
                        order=region.order,
                        font_size=region.font_size,
                        extras=extras,
                    )
                )
            regions_map[page_idx] = converted
        return regions_map

    def _resolve_page_sizes(self, pdf_path: Path) -> Dict[int, Tuple[float, float]]:
        sizes: Dict[int, Tuple[float, float]] = {}
        if fitz is not None:
            try:
                doc = fitz.open(str(pdf_path))
                for idx in range(doc.page_count):
                    rect = doc.load_page(idx).rect
                    sizes[idx] = (float(rect.width or 612.0), float(rect.height or 792.0))
                doc.close()
            except Exception:
                sizes = {}
        if not sizes:
            try:
                reader = PdfReader(str(pdf_path))
                for idx, page in enumerate(reader.pages):
                    width = float(page.mediabox.width or 612)
                    height = float(page.mediabox.height or 792)
                    sizes[idx] = (width, height)
            except Exception:
                pass
        return sizes

    def _surya_layout(self, page_count: int) -> Dict[int, List[LayoutRegion]]:
        detector = self._ensure_surya_detector()
        if self._page_image_store is None:
            raise RuntimeError("Surya layout backend requires a PageImageStore with rasterized pages.")
        regions_map: Dict[int, List[LayoutRegion]] = {}
        for page_idx in range(page_count):
            image_path = self._page_image_store.get_page_image(page_idx)
            if image_path is None:
                LOGGER.warning("Surya backend missing page raster for index %s", page_idx + 1)
                continue
            page_width, page_height = self._page_sizes.get(page_idx, (612.0, 792.0))
            try:
                detected = detector.detect(image_path)
            except Exception as exc:  # pragma: no cover - heavy dependency
                LOGGER.warning("Surya detection failed on page %s: %s", page_idx + 1, exc)
                continue
            converted: List[LayoutRegion] = []
            for order, region in enumerate(detected):
                polygon = list(region.polygon) if region.polygon else _bbox_to_polygon(region.bbox)
                extras = {
                    "layout_confidence": region.confidence,
                    "source": "surya",
                    "polygon": polygon,
                    "page_index": page_idx + 1,
                    "page_width_pt": page_width,
                    "page_height_pt": page_height,
                }
                converted.append(
                    LayoutRegion(
                        text="",
                        tag=region.label.lower(),
                        bbox=region.bbox,
                        column=1,
                        order=order,
                        font_size=0.0,
                        extras=extras,
                    )
                )
            regions_map[page_idx] = converted
        return regions_map

    def _ensure_surya_detector(self) -> SuryaLayoutDetector:
        if self._surya_detector is None:
            checkpoint_dir = MODELS_ROOT / SURYA_MODEL_SUBDIR
            self._surya_detector = SuryaLayoutDetector(
                checkpoint_dir=checkpoint_dir,
                enable_math_detector=self._surya_math_detector,
            )
        if not self._surya_detector.available():
            raise RuntimeError(
                "Surya backend requested but surya-ocr is not installed. Run `pip install surya-ocr` to continue."
            )
        return self._surya_detector

    def build_vision_branches(
        self,
        layout_segments: Dict[int, List[LayoutRegion]],
        page_store: PageImageStore | None = None,
    ) -> Dict[str, VisionSynthesisBranch]:
        if not self._vision_branch_enabled or not layout_segments:
            self._vision_branch_cache = {}
            return {}
        branches: Dict[str, VisionSynthesisBranch] = {}
        page_image_cache: Dict[int, str | None] = {}
        for page_idx, regions in layout_segments.items():
            page_image = None
            if page_store and page_store.enabled:
                if page_idx not in page_image_cache:
                    image_path = page_store.get_page_image(page_idx)
                    page_image_cache[page_idx] = str(image_path) if image_path else None
                page_image = page_image_cache[page_idx]
            for region in regions:
                extras = dict(region.extras or {})
                branch_id = extras.pop("branch_id", None)
                if not branch_id:
                    continue
                branch = VisionSynthesisBranch(
                    branch_id=branch_id,
                    page_index=page_idx + 1,
                    region_type=region.tag,
                    bbox=region.bbox,
                    extras=extras,
                    page_image=page_image,
                )
                branches[branch_id] = branch
        self._vision_branch_cache = branches
        return branches

    def vision_branches(self) -> Dict[str, VisionSynthesisBranch]:
        return dict(self._vision_branch_cache)

    def _predict_label(self, region: LayoutRegion) -> str:
        voter = EnsembleVoter(threshold=0.35)
        heuristic = self._heuristic_label(region)
        voter.add("heuristic", heuristic, score=0.6)
        if self._use_layoutlm and self._model is not None and self._processor is not None:
            refined = self._layoutlm_refine(region)
            if refined:
                label, confidence = refined
                voter.add("layoutlm", label, score=confidence, weight=1.2)
        return voter.best(default=heuristic)

    def _heuristic_label(self, region: LayoutRegion) -> str:
        tag = region.tag
        if tag == "heading":
            return "section-header"
        if tag == "question":
            return "question-stem"
        if tag == "answer":
            return "solution-text"
        if tag == "figure":
            return "figure"
        if tag == "table":
            return "table"
        if tag == "list":
            return "work-area"
        if tag == "formula":
            return "display-equation"
        if region.extras.get("header_level"):
            return "subsection-header"
        return "paragraph"

    def _layoutlm_refine(self, region: LayoutRegion) -> tuple[str, float] | None:  # pragma: no cover - heavy dependency
        if self._model is None or self._processor is None or self._device is None:
            return None
        tokens = region.text.split()
        if not tokens:
            return None
        truncated = tokens[:128]
        bbox = region.bbox or (0, 0, 1, 1)
        norm_box = self._normalize_bbox(bbox)
        boxes = [norm_box] * len(truncated)
        try:
            encoded = self._processor(
                text=[" ".join(truncated)],
                boxes=[boxes],
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=512,
            )
            encoded = {k: v.to(self._device) for k, v in encoded.items()}
            with torch.no_grad():
                outputs = self._model(**encoded)
            logits = outputs.logits[0]
            if F is None:
                return None
            probs = F.softmax(logits, dim=-1)
            pred_scores, pred_indices = probs.max(dim=-1)
            pred_id = int(pred_indices[0].item())
            confidence = float(pred_scores[0].item())
        except Exception as exc:
            LOGGER.debug("LayoutLMv3 inference failed: %s", exc)
            return None
        return self._label_map.get(pred_id, "paragraph"), confidence

    @staticmethod
    def _normalize_bbox(bbox: Tuple[float, float, float, float]) -> List[int]:
        x0, y0, x1, y1 = bbox
        width = max(1.0, x1 - x0)
        height = max(1.0, y1 - y0)
        return [
            int(max(0, min(1000, x0 / width * 1000))),
            int(max(0, min(1000, y0 / height * 1000))),
            int(max(0, min(1000, x1 / width * 1000))),
            int(max(0, min(1000, y1 / height * 1000))),
        ]

    def _resolve_device(self) -> "torch.device":
        if torch is None:
            raise RuntimeError("PyTorch is required for LayoutLM inference.")
        value = (LAYOUTLM_DEVICE_ENV or "cpu").strip().lower()
        if value == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            device = torch.device(value)
        except Exception:
            LOGGER.warning("Invalid LATEXIFY_LAYOUTLM_DEVICE '%s'; defaulting to CPU.", value)
            device = torch.device("cpu")
        if device.type == "cuda" and not torch.cuda.is_available():
            LOGGER.warning("LATEXIFY_LAYOUTLM_DEVICE set to CUDA but no GPU detected; using CPU.")
            device = torch.device("cpu")
        return device

    def close(self) -> None:
        self._layout.close()
        if self._model is not None and self._device is not None:
            try:
                if self._device.type == "cuda":
                    self._model.to("cpu")
                    torch.cuda.empty_cache()
            except Exception:
                pass
            self._model = None


class ClipCaptionVerifier:
    """Optional CLIP scorer that reorders OCR transcripts by visual alignment."""

    def __init__(self) -> None:
        self._model = None
        self._processor = None
        self._device = "cpu"
        self._available = False
        self._allow_download = os.environ.get("LATEXIFY_CLIP_LOCAL_ONLY", "0") != "1"
        if os.environ.get("LATEXIFY_ENABLE_CLIP_VERIFIER", "0") != "1":
            return
        try:  # pragma: no cover - optional heavy dependency
            import torch
            from transformers import CLIPModel, CLIPProcessor

            requested = CLIP_DEVICE_ENV.strip().lower()
            if requested == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            elif requested.startswith("cuda"):
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = "cpu"
            self._device = device
            kwargs = {"local_files_only": not self._allow_download}
            self._processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", **kwargs)
            self._model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", **kwargs)
            self._model.to(self._device)
            self._model.eval()
            self._torch = torch
            self._available = True
        except Exception as exc:  # pragma: no cover
            LOGGER.debug("CLIP verifier unavailable: %s", exc)

    def available(self) -> bool:
        return self._available and self._model is not None and self._processor is not None

    def score(self, image_path: str | None, caption: str) -> float:
        if not self.available() or not image_path or not caption.strip():
            return 0.0
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self._processor(text=[caption], images=[image], return_tensors="pt", padding=True)
            inputs = {k: v.to(self._device) if hasattr(v, "to") else v for k, v in inputs.items()}
            with self._torch.inference_mode():
                outputs = self._model(**inputs)
            logits = outputs.logits_per_image
            return float(logits[0][0].item())
        except Exception as exc:  # pragma: no cover
            LOGGER.debug("CLIP verifier scoring failed: %s", exc)
            return 0.0

    def rank_sources(self, image_path: str | None, sources: OrderedDict[str, str]) -> OrderedDict[str, str]:
        if not self.available() or len(sources) <= 1:
            return sources
        scored = []
        for backend, text in sources.items():
            scored.append((self.score(image_path, text), backend, text))
        scored.sort(key=lambda item: item[0], reverse=True)
        return OrderedDict((backend, text) for _, backend, text in scored)


@dataclass
class OCRResult:
    sources: OrderedDict[str, str]
    metadata: Dict[str, Dict[str, object]] = field(default_factory=dict)

    @property
    def backends(self) -> List[str]:
        return list(self.sources.keys())


TelemetryCallback = Callable[..., None]


def _available_system_memory_gb() -> float | None:
    if psutil is None:
        return None
    try:
        stats = psutil.virtual_memory()
    except Exception:  # pragma: no cover
        return None
    return stats.available / 1024**3


class OCRFallback:
    """OCR helper that can route between multiple local models."""

    def __init__(
        self,
        mode: str,
        cache_dir: Path,
        page_store: PageImageStore | None,
        models_dir: Path | None,
        telemetry: TelemetryCallback | None = None,
        math_page_hints: Sequence[bool] | None = None,
        quality_profile: QualityProfile | Dict[str, object] | None = None,
        monkey_adapter: MonkeyOCRAdapter | None = None,
    ) -> None:
        self.mode = mode
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.page_store = page_store
        self._quality_profile = quality_profile
        self.models_dir = models_dir or MODELS_ROOT
        self._nougat = None
        self._florence = None
        self._internvl = None
        self._qwenvl = None
        self._qwenvl_device_hint = _resolve_qwenvl_device()
        self._qwenvl_gpu_failures = 0
        self._mathvision = None
        self._mathocr_engine = None
        self._monkey_adapter = monkey_adapter
        self._monkey_cache: Dict[int, MonkeyOCRPageResult] = {}
        self._device_overrides: Dict[str, str] = {}
        self._availability = {
            "nougat": True,
            "florence2": True,
            "internvl": True,
            "qwenvl": True,
            "mathvision": True,
            "mathocr": MathOCREngine is not None,
            "pytesseract": pytesseract is not None,
            "monkeyocr": monkey_adapter is not None and monkey_adapter.available(),
        }
        self._download_attempted: Dict[str, bool] = {key: False for key in OCR_MODEL_SPECS}
        self._memory_requirements = {
            "florence2": 10 * 1024**3,  # ~10 GB
            "internvl": 22 * 1024**3,  # ~22 GB
            "qwenvl": 14 * 1024**3,  # ~14 GB
            "mathvision": 4 * 1024**3,
        }
        self._gpu_reservations: Dict[int, int] = defaultdict(int)
        self._gpu_preference = self._parse_gpu_preference(GPU_PREF_ENV)
        self._persistent_backends = os.environ.get("LATEXIFY_OCR_KEEP_LIVE", "0") == "1"
        self._sequential_mode = SEQUENTIAL_OCR
        self._sequential_cache: Dict[int, OCRResult] = {}
        self._active_pass_backend: str | None = None
        self._telemetry = telemetry
        self._vram_headroom_bytes = OCR_VRAM_HEADROOM_BYTES
        if FORCE_HEAVY_OCR:
            self._vram_headroom_bytes = 0
        hints = list(math_page_hints or [])
        self._math_page_hints = hints
        self._math_priority = any(hints)
        self._system_memory_floor_gb = max(0.0, SYSTEM_MEMORY_SKIP_HEAVY_GB)
        if quality_profile is not None:
            mode_hint = self._quality_mode()
            if mode_hint == "aggressive":
                self._math_priority = True
                self._sequential_mode = True
            elif mode_hint == "conservative":
                self._sequential_mode = False

    def _emit_telemetry(self, stage: str, status: str, **extra: Any) -> None:
        if not self._telemetry:
            return
        try:
            self._telemetry(stage, status, **extra)
        except Exception:
            # Telemetry is best-effort; never block OCR.
            pass

    def _system_memory_low(self) -> bool:
        available = _available_system_memory_gb()
        if available is None:
            return False
        return available < self._system_memory_floor_gb

    def _quality_mode(self) -> str:
        profile = self._quality_profile
        if isinstance(profile, QualityProfile):
            return profile.processing_mode
        if isinstance(profile, dict):
            return str(profile.get("processing_mode", "balanced"))
        return "balanced"

    def _page_requires_math(self, page_index: int) -> bool:
        if not self._math_page_hints:
            return False
        if page_index < 0 or page_index >= len(self._math_page_hints):
            return False
        return bool(self._math_page_hints[page_index])

    def _math_backends_allowed(self) -> bool:
        if FORCE_HEAVY_OCR:
            return True
        if self._system_memory_low():
            return False
        return self._availability.get("mathvision", True) or self._availability.get("mathocr", True)

    @staticmethod
    def _parse_gpu_preference(value: str | None) -> List[int]:
        if not value:
            return []
        prefs: List[int] = []
        for token in value.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                prefs.append(int(token))
            except ValueError:
                LOGGER.warning("Ignoring invalid GPU preference token '%s'", token)
        return prefs

    def _preferred_gpu_index(
        self, inventory: List[Tuple[int, int, int]], exclude: Set[int] | None = None
    ) -> int | None:
        if exclude:
            inventory = [entry for entry in inventory if entry[0] not in exclude]
        if not inventory:
            return None
        if self._gpu_preference:
            preferred_lookup = {idx for idx, _, _ in inventory}
            for candidate in self._gpu_preference:
                if candidate in preferred_lookup:
                    return candidate
        return inventory[0][0]

    def _ensure_backend_weights(self, backend: str) -> Path:
        spec = OCR_MODEL_SPECS.get(backend)
        if spec is None:
            return self.models_dir
        target = self.models_dir / spec["subpath"]
        if target.exists() and any(target.iterdir()):
            return target
        if snapshot_download is None:
            if self._availability.get(backend, True):
                LOGGER.warning(
                    "huggingface_hub unavailable; cannot auto-download %s. Install huggingface_hub or "
                    "pre-populate %s.",
                    spec["repo_id"],
                    target,
                )
                self._availability[backend] = False
            return target
        if self._download_attempted.get(backend):
            return target
        self._download_attempted[backend] = True
        target.mkdir(parents=True, exist_ok=True)
        try:
            LOGGER.info("Auto-downloading %s into %s", spec["repo_id"], target)
            snapshot_download(
                spec["repo_id"],
                local_dir=str(target),
                local_dir_use_symlinks=False,
                resume_download=True,
            )
        except Exception as exc:  # pragma: no cover - depends on env
            LOGGER.warning(
                "Failed to download %s (%s). Please run `huggingface-cli login` and ensure access.",
                spec["repo_id"],
                exc,
            )
        return target

    def _ensure_python_packages(self, backend: str) -> bool:
        missing = [
            pkg
            for pkg in BACKEND_PYTHON_DEPS.get(backend, ())
            if importlib.util.find_spec(pkg) is None
        ]
        if missing:
            if self._availability.get(backend, True):
                LOGGER.warning(
                    "%s backend requires Python packages missing locally: %s. "
                    "Install them via `pip install %s` inside the release venv.",
                    backend,
                    ", ".join(missing),
                    " ".join(missing),
                )
                self._availability[backend] = False
            return False
        return True

    def _cuda_inventory(self) -> List[Tuple[int, int, int]]:
        if torch is None or not torch.cuda.is_available():
            return []
        inventory: List[Tuple[int, int, int]] = []
        for idx in range(torch.cuda.device_count()):
            try:
                with torch.cuda.device(idx):
                    free_bytes, total_bytes = torch.cuda.mem_get_info()
            except Exception:  # pragma: no cover - driver variance
                try:
                    total_bytes = torch.cuda.get_device_properties(idx).total_memory
                except Exception as exc:
                    LOGGER.warning("Unable to inspect CUDA device %s: %s", idx, exc)
                    continue
                free_bytes = total_bytes
            except Exception as exc:  # pragma: no cover - driver variance
                LOGGER.warning("Unable to inspect CUDA device %s: %s", idx, exc)
                continue
            inventory.append((idx, total_bytes, free_bytes))
        if not inventory:
            return []
        inventory.sort(key=lambda item: (-item[2], -item[1], item[0]))
        if self._gpu_preference:
            order = {dev: pos for pos, dev in enumerate(self._gpu_preference)}
            inventory.sort(key=lambda item: (order.get(item[0], len(order)), -item[2], -item[1], item[0]))
        return inventory

    def _select_backend_device(self, backend: str, exclude: set[int] | None = None) -> tuple[str, int | None]:
        cached = self._device_overrides.get(backend)
        if cached:
            if cached.startswith("cuda:"):
                return cached, int(cached.split(":")[1])
            return cached, None
        override_device = FORCE_GPU_OCR_DEVICE if backend in FORCE_GPU_OCR_BACKENDS else None
        if torch is None:
            raise RuntimeError("PyTorch is required for OCR backends but is unavailable.")
        if override_device:
            if not torch.cuda.is_available():
                LOGGER.warning(
                    "LATEXIFY_FORCE_GPU_OCR=%s but CUDA is unavailable; ignoring override.",
                    override_device,
                )
            else:
                idx_override: int | None = None
                if override_device.startswith("cuda:"):
                    try:
                        idx_override = int(override_device.split(":")[1])
                    except ValueError:
                        idx_override = None
                if idx_override is not None and idx_override >= torch.cuda.device_count():
                    LOGGER.warning(
                        "LATEXIFY_FORCE_GPU_OCR requested %s but only %s CUDA device(s) detected; "
                        "falling back to automatic placement.",
                        override_device,
                        torch.cuda.device_count(),
                    )
                else:
                    LOGGER.warning(
                        "LATEXIFY_FORCE_GPU_OCR active; forcing %s backend onto %s and bypassing VRAM heuristics.",
                        backend,
                        override_device,
                    )
                    return override_device, idx_override
        if not torch.cuda.is_available():
            LOGGER.warning(
                "%s backend requested but CUDA is unavailable; running on CPU. Expect slower throughput.",
                backend,
            )
            return "cpu", None
        inventory = self._cuda_inventory()
        if not inventory:
            LOGGER.warning(
                "CUDA runtime detected but no devices enumerated; routing %s backend to CPU.",
                backend,
            )
            return "cpu", None
        normalized_backend = backend.lower()
        if FORCE_GPU_BACKEND_OVERRIDES and (
            "all" in FORCE_GPU_BACKEND_OVERRIDES or normalized_backend in FORCE_GPU_BACKEND_OVERRIDES
        ):
            idx_override = self._preferred_gpu_index(inventory, exclude)
            if idx_override is not None:
                LOGGER.warning(
                    "LATEXIFY_OCR_FORCE_GPU forcing %s backend onto cuda:%s; bypassing VRAM heuristics.",
                    backend,
                    idx_override,
                )
                return f"cuda:{idx_override}", idx_override
        requirement = self._memory_requirements.get(backend)
        headroom_bytes = self._vram_headroom_bytes

        def _pick_device(inventory_entries: List[Tuple[int, int, int]]) -> int | None:
            for idx, total, free in inventory_entries:
                if exclude and idx in exclude:
                    continue
                reserved = self._gpu_reservations.get(idx, 0)
                available = free - reserved if not FORCE_HEAVY_OCR else total - reserved
                if not FORCE_HEAVY_OCR and headroom_bytes:
                    available -= headroom_bytes
                if available <= 0:
                    continue
                if requirement is None or FORCE_HEAVY_OCR or available >= requirement:
                    return idx
            return None

        picked_idx = _pick_device(inventory)
        if picked_idx is None and requirement and not FORCE_HEAVY_OCR:
            self._flush_cuda_cache()
            inventory = self._cuda_inventory()
            picked_idx = _pick_device(inventory)
        if picked_idx is None:
            if requirement and not FORCE_HEAVY_OCR:
                needed = requirement / 1024**3
                reason = "insufficient free memory"
                if self._vram_headroom_bytes:
                    reason += f" (applying {self._vram_headroom_bytes / 1024**3:.1f} GiB headroom)"
                LOGGER.warning(
                    "No GPU reports sufficient free memory for %s (needs ≈%.1f GB); %s. "
                    "Routing backend to CPU/offload; expect slower throughput.",
                    backend,
                    needed,
                    reason,
                )
                return "cpu", None
            picked_idx = inventory[0][0]
            needed = requirement / 1024**3 if requirement else 0
            LOGGER.warning(
                "No GPU reports sufficient free memory for %s (needs ≈%.1f GB). "
                "Forcing allocation on cuda:%s; expect longer initialization.",
                backend,
                needed,
                picked_idx,
            )
        return f"cuda:{picked_idx}", picked_idx

    def _flush_cuda_cache(self) -> None:
        if torch is None or not torch.cuda.is_available():
            return
        try:
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
        except Exception:
            LOGGER.debug("Unable to flush CUDA allocator caches.", exc_info=True)

    def _wait_for_vram_budget(self, backend: str, timeout: float = 10.0, interval: float = 0.5) -> None:
        if torch is None or not torch.cuda.is_available():
            return
        requirement = self._memory_requirements.get(backend)
        if not requirement:
            return
        deadline = time.monotonic() + max(0.0, timeout)
        while time.monotonic() < deadline:
            inventory = self._cuda_inventory()
            if not inventory:
                return
            for idx, total, free in inventory:
                reserved = self._gpu_reservations.get(idx, 0)
                available = free - reserved if not FORCE_HEAVY_OCR else total - reserved
                if not FORCE_HEAVY_OCR and self._vram_headroom_bytes:
                    available -= self._vram_headroom_bytes
                if available >= requirement:
                    return
            with contextlib.suppress(Exception):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            time.sleep(max(0.05, interval))
        LOGGER.debug(
            "Timed out waiting for ≈%.1f GiB of free VRAM for %s after %.1f seconds.",
            requirement / 1024**3,
            backend,
            timeout,
        )

    def _reserve_gpu(self, idx: int | None, backend: str) -> None:
        if idx is None:
            return
        requirement = self._memory_requirements.get(backend)
        if requirement:
            self._gpu_reservations[idx] += requirement

    def _free_gpu(self, idx: int | None, backend: str) -> None:
        if idx is None:
            return
        requirement = self._memory_requirements.get(backend)
        if not requirement:
            return
        current = self._gpu_reservations.get(idx, 0)
        self._gpu_reservations[idx] = max(0, current - requirement)

    def _teardown_adapter(self, adapter: object | None, backend: str) -> None:
        if adapter is None:
            return
        close = getattr(adapter, "close", None)
        if callable(close):
            try:
                LOGGER.info("Releasing %s backend and clearing CUDA cache.", backend)
                close()
            except Exception:
                LOGGER.debug("Failed to close backend %s cleanly.", backend, exc_info=True)

    def _release_backend(self, backend: str) -> None:
        released = False
        device = self._device_overrides.pop(backend, None)
        if backend == "florence2" and self._florence is not None:
            self._teardown_adapter(self._florence, "florence2")
            self._florence = None
            released = True
        if backend == "internvl" and self._internvl is not None:
            self._teardown_adapter(self._internvl, "internvl")
            self._internvl = None
            released = True
            if _release_shared_adapter is not None:
                try:
                    _release_shared_adapter("internvl")
                except Exception:
                    LOGGER.debug("Failed to release shared InternVL adapter.", exc_info=True)
        if backend == "qwenvl" and self._qwenvl is not None:
            try:
                close = getattr(self._qwenvl, "close", None)
                if callable(close):
                    close()
            except Exception:
                LOGGER.debug("Failed to close Qwen-VL adapter cleanly.", exc_info=True)
            self._qwenvl = None
            released = True
        if backend == "mathvision" and self._mathvision is not None:
            self._mathvision = None
            released = True
        if backend == "mathocr" and self._mathocr_engine is not None:
            self._mathocr_engine = None
            released = True
        idx = None
        if device and device.startswith("cuda:"):
            try:
                idx = int(device.split(":")[1])
            except ValueError:
                idx = None
        if released:
            self._free_gpu(idx, backend)
        if released and torch is not None and torch.cuda.is_available():  # pragma: no cover
            try:
                if device and device.startswith("cuda:"):
                    current = torch.cuda.current_device()
                    target = int(device.split(":")[1])
                    torch.cuda.set_device(target)
                    torch.cuda.empty_cache()
                    torch.cuda.set_device(current)
                else:
                    torch.cuda.empty_cache()
            except Exception:
                pass

    def _should_defer_release(self, backend: str) -> bool:
        return self._sequential_mode and self._active_pass_backend == backend

    def _attempt_order(self) -> List[str]:
        if self.mode == "auto":
            order: List[str] = []
            mode_hint = self._quality_mode()
            math_focus = self._math_priority or mode_hint == "aggressive"
            math_allowed = math_focus and self._math_backends_allowed()
            heavy_allowed = FORCE_HEAVY_OCR or not self._system_memory_low()
            if mode_hint == "conservative":
                heavy_allowed = False
            elif mode_hint == "aggressive":
                heavy_allowed = True
            if self._availability.get("monkeyocr", False):
                order.append("monkeyocr")
            if heavy_allowed:
                if self._availability.get("florence2", True):
                    order.append("florence2")
                if PREFER_QWEN_VL and self._availability.get("qwenvl", True):
                    order.append("qwenvl")
                elif self._availability.get("internvl", True):
                    order.append("internvl")
                elif self._availability.get("qwenvl", True):
                    order.append("qwenvl")
                LOGGER.debug("Prioritizing Florence2 before other heavy OCR backends.")
            if not heavy_allowed:
                available = _available_system_memory_gb()
                if available is not None:
                    LOGGER.info(
                        "Skipping Florence/InternVL due to limited system memory (≈%.1f GiB free).",
                        available,
                    )
            order.append("nougat")
            if math_allowed and self._availability.get("mathvision", True):
                order.append("mathvision")
            if math_allowed and self._availability.get("mathocr", True):
                order.append("mathocr")
            if pytesseract is not None and self._availability.get("pytesseract", True):
                if mode_hint == "aggressive" and "pytesseract" not in order:
                    order.insert(1, "pytesseract")
                elif "pytesseract" not in order:
                    order.append("pytesseract")
            return order
        mapping = {
            "nougat": ["nougat"],
            "mathvision": ["mathvision"],
            "mathocr": ["mathocr"],
            "florence2": ["florence2"],
            "internvl": ["internvl"],
            "qwenvl": ["qwenvl"],
            "pytesseract": ["pytesseract"],
            "monkeyocr": ["monkeyocr"],
        }
        return mapping.get(self.mode, [])

    def _should_run_backend(self, backend: str, page_index: int) -> bool:
        if backend == "monkeyocr":
            return self._availability.get("monkeyocr", False)
        if backend in {"mathvision", "mathocr"}:
            return self._math_priority and self._math_backends_allowed() and self._page_requires_math(page_index)
        if backend in {"florence2", "internvl", "qwenvl"} and not FORCE_HEAVY_OCR and self._system_memory_low():
            return False
        return True

    def _invoke_backend(self, backend: str, page_index: int) -> tuple[str, Dict[str, object] | None] | None:
        handlers = {
            "nougat": self._run_nougat,
            "mathvision": self._run_mathvision,
            "mathocr": self._run_mathocr,
            "florence2": self._run_florence,
            "internvl": self._run_internvl,
             "qwenvl": self._run_qwenvl,
            "pytesseract": self._run_tesseract,
            "monkeyocr": self._run_monkey,
        }
        handler = handlers.get(backend)
        if handler is None:
            return None
        result = handler(page_index)
        if result is None:
            return None
        if isinstance(result, tuple):
            return result
        return result, None

    def _run_monkey(self, page_index: int) -> str | None:
        if not self._monkey_adapter:
            return None
        if page_index in self._monkey_cache:
            return self._monkey_cache[page_index].text
        try:
            result = self._monkey_adapter.analyze_page(page_index)
        except Exception as exc:
            LOGGER.warning("MonkeyOCR failed on page %s: %s", page_index + 1, exc)
            self._availability["monkeyocr"] = False
            return None
        self._monkey_cache[page_index] = result
        return result.text

    def prepare_document(self, total_pages: int) -> None:
        if not self._sequential_mode or total_pages <= 0 or self._sequential_cache:
            return
        LOGGER.info("Sequential OCR mode active; processing %s page(s) per backend.", total_pages)
        order = self._attempt_order()
        page_sources: List[OrderedDict[str, str]] = [OrderedDict() for _ in range(total_pages)]
        page_metadata: List[Dict[str, Dict[str, object]]] = [{} for _ in range(total_pages)]
        for backend in order:
            self._active_pass_backend = backend
            LOGGER.info("[ocr] Backend %s pass started.", backend)
            self._emit_telemetry(
                f"ocr/{backend}",
                "started",
                backend=backend,
                notes=f"{backend} pass started ({total_pages} pages)",
                pages=total_pages,
            )
            for page_index in range(total_pages):
                if not self._should_run_backend(backend, page_index):
                    continue
                result = self._invoke_backend(backend, page_index)
                if not result:
                    continue
                text, meta = result
                if text:
                    page_sources[page_index][backend] = text
                    if meta:
                        page_metadata[page_index][backend] = meta
            if backend in {"florence2", "internvl", "qwenvl", "mathvision", "mathocr"}:
                self._release_backend(backend)
                if torch is not None and torch.cuda.is_available():  # pragma: no cover
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
            LOGGER.info(
                "[ocr] Backend %s pass complete (%s/%s pages captured).",
                backend,
                sum(1 for sources in page_sources if backend in sources),
                total_pages,
            )
            captured = sum(1 for sources in page_sources if backend in sources)
            self._emit_telemetry(
                f"ocr/{backend}",
                "completed",
                backend=backend,
                pages=total_pages,
                captured_pages=captured,
                notes=f"{backend} captured {captured}/{total_pages} pages",
            )
        self._active_pass_backend = None
        for page_index, sources in enumerate(page_sources):
            md = page_metadata[page_index] if page_metadata else {}
            self._sequential_cache[page_index] = OCRResult(sources, md)

    def _lazy_nougat(self):
        if self._nougat is not None:
            return self._nougat
        model_dir = self._ensure_backend_weights("nougat")
        model_dir = model_dir if model_dir else self.models_dir / "ocr" / "nougat-small"
        if not model_dir.exists() or not any(model_dir.iterdir()):
            if self._availability.get("nougat", True):
                LOGGER.warning("Nougat model directory missing: %s", model_dir)
                self._availability["nougat"] = False
            return None
        try:
            from ..models.nougat_adapter import NougatAdapter, NougatAdapterConfig
        except Exception as exc:  # pragma: no cover - optional import
            LOGGER.warning("Failed to import Nougat adapter: %s", exc)
            if self._availability.get("nougat", True):
                LOGGER.warning("Failed to import Nougat adapter: %s", exc)
                self._availability["nougat"] = False
            return None
        try:
            pix2tex_dir = self.models_dir / "ocr" / "pix2tex-base"
            self._nougat = NougatAdapter(
                NougatAdapterConfig(
                    model_dir=model_dir,
                    fallback_pix2tex_dir=pix2tex_dir if pix2tex_dir.exists() else None,
                )
            )
        except Exception as exc:  # pragma: no cover - heavy deps
            LOGGER.warning("Failed to initialize Nougat: %s", exc)
            self._nougat = None
            self._availability["nougat"] = False
        return self._nougat

    def _lazy_florence(self):
        if self._florence is not None:
            return self._florence
        if not self._availability.get("florence2", True):
            return None
        if not self._ensure_python_packages("florence2"):
            return None
        # Make sure InternVL releases its VRAM before Florence loads to avoid OOM churn.
        self._release_backend("internvl")
        if torch is not None and torch.cuda.is_available():  # pragma: no cover - device specific
            with contextlib.suppress(Exception):
                torch.cuda.empty_cache()
        self._wait_for_vram_budget("florence2")
        try:
            from ..models.model_adapters import FlorenceAdapter, FlorenceConfig
        except Exception as exc:  # pragma: no cover
            if self._availability.get("florence2", True):
                LOGGER.warning("Failed to import Florence adapter: %s", exc)
                self._availability["florence2"] = False
            return None
        model_dir = self._ensure_backend_weights("florence2")
        if not model_dir.exists() or not any(model_dir.iterdir()):
            if self._availability.get("florence2", True):
                LOGGER.warning("Florence model directory missing: %s", model_dir)
                self._availability["florence2"] = False
            return None
        exclude: set[int] = set()
        while True:
            device, device_idx = self._select_backend_device("florence2", exclude)
            self._emit_telemetry(
                "ocr_load/florence2",
                "started",
                backend="florence2",
                device=device,
                device_index=device_idx,
                notes=f"Loading Florence2 on {device}",
            )
            try:
                self._florence = FlorenceAdapter(FlorenceConfig(model_dir=model_dir, device=device))
                self._device_overrides["florence2"] = device
                self._reserve_gpu(device_idx, "florence2")
                LOGGER.info("florence2 backend assigned to %s", device)
                self._emit_telemetry(
                    "ocr_load/florence2",
                    "completed",
                    backend="florence2",
                    device=device,
                    device_index=device_idx,
                )
                return self._florence
            except RuntimeError as exc:
                if "CUDA out of memory" in str(exc) and device_idx is not None:
                    self._emit_telemetry(
                        "ocr_load/florence2",
                        "retry",
                        backend="florence2",
                        device=device,
                        device_index=device_idx,
                        notes="CUDA out of memory",
                    )
                    exclude.add(device_idx)
                    continue
                LOGGER.warning("Failed to initialize Florence model: %s", exc)
                self._florence = None
                self._availability["florence2"] = False
                self._device_overrides.pop("florence2", None)
                self._emit_telemetry(
                    "ocr_load/florence2",
                    "failed",
                    backend="florence2",
                    device=device,
                    device_index=device_idx,
                    notes=str(exc),
                )
                return None

    def _lazy_internvl(self):
        if self._internvl is not None:
            return self._internvl
        if not self._availability.get("internvl", True):
            return None
        if not self._ensure_python_packages("internvl"):
            return None
        # Florence2 holds onto substantial VRAM; release it explicitly before loading InternVL.
        self._release_backend("florence2")
        if torch is not None and torch.cuda.is_available():  # pragma: no cover - device specific
            with contextlib.suppress(Exception):
                torch.cuda.empty_cache()
        self._wait_for_vram_budget("internvl")
        try:
            from ..models.model_adapters import (
                InternVLAdapter,
                InternVLConfig,
                get_shared_adapter,
                register_shared_adapter,
            )
        except Exception as exc:  # pragma: no cover
            if self._availability.get("internvl", True):
                LOGGER.warning("Failed to import InternVL adapter: %s", exc)
                self._availability["internvl"] = False
            return None
        model_dir = self._ensure_backend_weights("internvl")
        if not model_dir.exists() or not any(model_dir.iterdir()):
            if self._availability.get("internvl", True):
                LOGGER.warning("InternVL model directory missing: %s", model_dir)
                self._availability["internvl"] = False
            return None
        shared = None
        try:
            candidate = get_shared_adapter("internvl")
            if isinstance(candidate, InternVLAdapter):
                shared = candidate
        except Exception:
            shared = None
        if shared is not None:
            self._internvl = shared
            LOGGER.info("Reusing shared InternVL adapter for OCR.")
            return self._internvl
        exclude: set[int] = set()
        while True:
            device, device_idx = self._select_backend_device("internvl", exclude)
            self._emit_telemetry(
                "ocr_load/internvl",
                "started",
                backend="internvl",
                device=device,
                device_index=device_idx,
                notes=f"Loading InternVL on {device}",
            )
            try:
                self._internvl = InternVLAdapter(InternVLConfig(model_dir=model_dir, device=device))
                self._device_overrides["internvl"] = device
                self._reserve_gpu(device_idx, "internvl")
                try:
                    register_shared_adapter("internvl", self._internvl)
                except Exception:
                    LOGGER.debug("Unable to register shared InternVL adapter.", exc_info=True)
                LOGGER.info("internvl backend assigned to %s", device)
                self._emit_telemetry(
                    "ocr_load/internvl",
                    "completed",
                    backend="internvl",
                    device=device,
                    device_index=device_idx,
                )
                return self._internvl
            except RuntimeError as exc:
                if "CUDA out of memory" in str(exc) and device_idx is not None:
                    self._emit_telemetry(
                        "ocr_load/internvl",
                        "retry",
                        backend="internvl",
                        device=device,
                        device_index=device_idx,
                        notes="CUDA out of memory",
                    )
                    exclude.add(device_idx)
                    continue
                LOGGER.warning("Failed to initialize InternVL: %s", exc)
                self._internvl = None
                self._availability["internvl"] = False
                self._device_overrides.pop("internvl", None)
                self._emit_telemetry(
                    "ocr_load/internvl",
                    "failed",
                    backend="internvl",
                    device=device,
                    device_index=device_idx,
                    notes=str(exc),
                )
                return None

    def _lazy_qwenvl(self):
        if self._qwenvl is not None:
            return self._qwenvl
        if not self._availability.get("qwenvl", True):
            return None
        if not self._ensure_python_packages("qwenvl"):
            return None
        self._release_backend("florence2")
        self._release_backend("internvl")
        device_hint = getattr(self, "_qwenvl_device_hint", _resolve_qwenvl_device())
        prefer_gpu = device_hint.startswith("cuda")
        if prefer_gpu and torch is not None and torch.cuda.is_available():  # pragma: no cover
            with contextlib.suppress(Exception):
                torch.cuda.empty_cache()
            self._wait_for_vram_budget("qwenvl")
        model_dir = self._ensure_backend_weights("qwenvl")
        if not model_dir.exists() or not any(model_dir.iterdir()):
            if self._availability.get("qwenvl", True):
                LOGGER.warning("Qwen-VL model directory missing: %s", model_dir)
                self._availability["qwenvl"] = False
            return None
        exclude: set[int] = set()
        while True:
            device = device_hint
            device_idx = None
            if device == "auto":
                resolved, device_idx = self._select_backend_device("qwenvl", exclude)
                device = resolved or "cpu"
            elif device.startswith("cuda"):
                try:
                    device_idx = int(device.split(":")[1])
                except (IndexError, ValueError):
                    device_idx = 0 if torch and torch.cuda.is_available() else None
            self._emit_telemetry(
                "ocr_load/qwenvl",
                "started",
                backend="qwenvl",
                device=device,
                device_index=device_idx,
                notes=f"Loading Qwen-VL on {device}",
            )
            try:
                adapter = get_vlm_adapter(
                    "qwen-vl",
                    prompt=QWEN_VL_PROMPT,
                    max_new_tokens=QWEN_VL_MAX_NEW_TOKENS,
                    temperature=QWEN_VL_TEMPERATURE,
                    top_p=QWEN_VL_TOP_P,
                    device=device,
                    model_id=str(model_dir),
                    load_in_8bit=QWEN_VL_LOAD_IN_8BIT,
                    load_in_4bit=QWEN_VL_LOAD_IN_4BIT,
                )
                self._qwenvl = adapter
                self._qwenvl_device_hint = device
                self._device_overrides["qwenvl"] = device
                self._reserve_gpu(device_idx, "qwenvl")
                LOGGER.info("qwenvl backend assigned to %s", device)
                self._emit_telemetry(
                    "ocr_load/qwenvl",
                    "completed",
                    backend="qwenvl",
                    device=device,
                    device_index=device_idx,
                )
                return self._qwenvl
            except RuntimeError as exc:
                message = str(exc)
                self._emit_telemetry(
                    "ocr_load/qwenvl",
                    "retry" if "CUDA out of memory" in message else "failed",
                    backend="qwenvl",
                    device=device,
                    device_index=device_idx,
                    notes=message,
                )
                if "CUDA out of memory" in message and device.startswith("cuda"):
                    LOGGER.warning(
                        "Qwen-VL OOM on %s; retrying with a different device or CPU fallback.",
                        device,
                    )
                    self._qwenvl_gpu_failures += 1
                    if self._qwenvl_gpu_failures >= QWEN_VL_MAX_GPU_RETRIES:
                        LOGGER.warning(
                            "Exceeded %s Qwen-VL GPU retries; forcing CPU/offload execution.",
                            QWEN_VL_MAX_GPU_RETRIES,
                        )
                        self._qwenvl_device_hint = "cpu"
                        device_hint = "cpu"
                    else:
                        exclude.add(device_idx if device_idx is not None else 0)
                    continue
                LOGGER.warning("Failed to initialize Qwen-VL adapter: %s", exc)
                self._availability["qwenvl"] = False
                self._device_overrides.pop("qwenvl", None)
                return None

    def _lazy_mathvision(self):
        if self._mathvision is not None:
            return self._mathvision
        if not self._availability.get("mathvision", True):
            return None
        try:
            from ..models.model_adapters import MathVisionAdapter, MathVisionConfig
        except Exception as exc:  # pragma: no cover
            if self._availability.get("mathvision", True):
                LOGGER.warning("Failed to import MathVision adapter: %s", exc)
                self._availability["mathvision"] = False
            return None
        model_dir = self._ensure_backend_weights("mathvision")
        if not model_dir.exists() or not any(model_dir.iterdir()):
            LOGGER.warning("MathVision model directory missing: %s", model_dir)
            self._availability["mathvision"] = False
            return None
        try:
            self._emit_telemetry("ocr_load/mathvision", "started", backend="mathvision", notes="Loading MathVision.")
            self._mathvision = MathVisionAdapter(MathVisionConfig(model_dir=model_dir))
            LOGGER.info("mathvision backend initialized for math-heavy OCR.")
            self._emit_telemetry("ocr_load/mathvision", "completed", backend="mathvision")
            return self._mathvision
        except RuntimeError as exc:
            LOGGER.warning("Failed to initialize MathVision model: %s", exc)
            self._availability["mathvision"] = False
            self._mathvision = None
            self._emit_telemetry("ocr_load/mathvision", "failed", backend="mathvision", notes=str(exc))
            return None

    def _page_image(self, page_index: int) -> Path | None:
        if self.page_store is None:
            return None
        return self.page_store.get_page_image(page_index)

    def _run_nougat(self, page_index: int) -> tuple[str, Dict[str, object]] | None:
        nougat = self._lazy_nougat()
        if nougat is None:
            return None
        image = self._page_image(page_index)
        if not image:
            return None
        try:
            text, confidence = nougat.predict_with_confidence(image)
            return text, {"confidence": confidence}
        except Exception as exc:  # pragma: no cover - model runtime
            LOGGER.warning("Nougat inference failed on page %s: %s", page_index + 1, exc)
            return None

    def _run_florence(self, page_index: int) -> str | None:
        florence = self._lazy_florence()
        if florence is None:
            return None
        image = self._page_image(page_index)
        if not image:
            return None
        try:
            return florence.predict(image)
        except Exception as exc:
            LOGGER.warning("Florence inference failed on page %s: %s", page_index + 1, exc)
            return None
        finally:
            if RELEASE_HEAVY_MODE == "page" and not self._should_defer_release("florence2"):
                self._release_backend("florence2")

    def _run_internvl(self, page_index: int) -> str | None:
        adapter = self._lazy_internvl()
        if adapter is None:
            return None
        image = self._page_image(page_index)
        if not image:
            return None
        try:
            return adapter.predict(image)
        except Exception as exc:
            LOGGER.warning("InternVL inference failed on page %s: %s", page_index + 1, exc)
            return None
        finally:
            if RELEASE_HEAVY_MODE == "page" and not self._should_defer_release("internvl"):
                self._release_backend("internvl")

    def _run_qwenvl(self, page_index: int) -> str | None:
        adapter = self._lazy_qwenvl()
        if adapter is None:
            return None
        image = self._page_image(page_index)
        if not image:
            return None
        try:
            return adapter.describe(Path(image), prompt=QWEN_VL_PROMPT)
        except Exception as exc:
            LOGGER.warning("Qwen-VL inference failed on page %s: %s", page_index + 1, exc)
            return None
        finally:
            if RELEASE_HEAVY_MODE == "page" and not self._should_defer_release("qwenvl"):
                self._release_backend("qwenvl")

    def _run_tesseract(self, page_index: int) -> str | None:
        if pytesseract is None:
            return None
        image = self._page_image(page_index)
        if not image:
            return None
        try:
            pil_image = Image.open(image).convert('RGB')
        except Exception as exc:
            LOGGER.warning("Failed to open page raster %s for pytesseract: %s", image, exc)
            return None
        text = pytesseract.image_to_string(pil_image)
        cleaned = text.strip()
        return cleaned or None

    def _run_mathvision(self, page_index: int) -> str | None:
        adapter = self._lazy_mathvision()
        if adapter is None:
            return None
        image = self._page_image(page_index)
        if not image:
            return None
        try:
            return adapter.predict(image)
        except Exception as exc:
            LOGGER.warning("MathVision inference failed on page %s: %s", page_index + 1, exc)
            return None

    def _run_mathocr(self, page_index: int) -> str | None:
        engine = self._lazy_mathocr()
        if engine is None:
            return None
        image = self._page_image(page_index)
        if not image:
            return None
        try:
            return engine.process_page(Path(image))
        except Exception as exc:
            LOGGER.warning("MathOCR inference failed on page %s: %s", page_index + 1, exc)
            return None

    def _lazy_mathocr(self):
        if self._mathocr_engine is not None:
            return self._mathocr_engine
        if not self._availability.get("mathocr", True):
            return None
        if MathOCREngine is None:
            self._availability["mathocr"] = False
            return None
        try:
            self._mathocr_engine = MathOCREngine(semantic_validation=MATHOCR_SEMANTIC_VALIDATION)
            return self._mathocr_engine
        except Exception as exc:  # pragma: no cover - env specific
            LOGGER.warning("Failed to initialize MathOCR engine: %s", exc)
            self._availability["mathocr"] = False
            self._mathocr_engine = None
            return None

    def shutdown(self) -> None:
        if RELEASE_HEAVY_MODE in {"page", "run"}:
            self._release_backend("florence2")
            self._release_backend("internvl")
            self._release_backend("mathvision")
            self._release_backend("mathocr")

    def extract(self, page_index: int) -> OCRResult:
        if self._sequential_mode and self._sequential_cache:
            cached = self._sequential_cache.get(page_index)
            if cached is not None and cached.sources:
                return cached
        attempt_order = self._attempt_order()
        sources: OrderedDict[str, str] = OrderedDict()
        metadata: Dict[str, Dict[str, object]] = {}
        try:
            for backend in attempt_order:
                if not self._should_run_backend(backend, page_index):
                    continue
                result = self._invoke_backend(backend, page_index)
                if not result:
                    continue
                text, meta = result
                if text:
                    cleaned = text.strip()
                    if cleaned:
                        sources[backend] = cleaned
                        if meta:
                            metadata[backend] = meta
        finally:
            if RELEASE_HEAVY_MODE == "page" and not self._persistent_backends:
                if not self._should_defer_release("florence2"):
                    self._release_backend("florence2")
                if not self._should_defer_release("internvl"):
                    self._release_backend("internvl")
                if not self._should_defer_release("mathvision"):
                    self._release_backend("mathvision")
                if not self._should_defer_release("mathocr"):
                    self._release_backend("mathocr")
        return OCRResult(sources, metadata)


def read_pdf_text(pdf_path: Path) -> List[str]:
    reader = PdfReader(str(pdf_path))
    texts: List[str] = []
    for page_idx, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # pragma: no cover - rare PyPDF failure
            LOGGER.warning("Failed to extract text for page %s: %s", page_idx, exc)
            text = ""
        texts.append(text)
    return texts


def run_pdfimages(pdf_path: Path, image_dir: Path) -> List[Path]:
    image_dir.mkdir(parents=True, exist_ok=True)
    prefix = image_dir / "asset"
    cmd = ["pdfimages", "-png", str(pdf_path), str(prefix)]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except FileNotFoundError:  # pragma: no cover - depends on host
        LOGGER.warning("pdfimages binary not found; image extraction skipped")
        return []
    except subprocess.CalledProcessError as exc:  # pragma: no cover - logged for debugging
        LOGGER.warning("pdfimages failed: %s", exc.stderr)
    images = sorted(image_dir.glob("asset*.png"))
    return images


def assign_images_to_pages(images: Sequence[Path], num_pages: int) -> Dict[int, List[str]]:
    if not images or num_pages == 0:
        return {idx: [] for idx in range(num_pages)}
    per_page = max(1, math.ceil(len(images) / max(1, num_pages)))
    mapping: Dict[int, List[str]] = {idx: [] for idx in range(num_pages)}
    for idx, image in enumerate(images):
        page_idx = min(num_pages - 1, idx // per_page)
        mapping[page_idx].append(str(image))
    return mapping


def detect_header_level(line: str) -> int:
    if not line:
        return 0
    normalized = line.strip()
    lower = normalized.lower()
    if any(lower.startswith(keyword) for keyword in HEADER_KEYWORDS):
        return 1 if lower.startswith("chapter") else 2
    if normalized.isupper() and len(normalized.split()) <= 8:
        return 1
    if len(normalized.split()) <= 10 and normalized.endswith(":"):
        return 2
    return 0


def extract_table_signature(lines: List[str]) -> Dict[str, int]:
    rows = []
    max_cols = 0
    for line in lines:
        if "|" in line:
            cells = [cell for cell in line.split("|") if cell.strip()]
            if cells:
                rows.append(line)
                max_cols = max(max_cols, len(cells))
    return {"rows": len(rows), "columns": max_cols or 1}


def estimate_list_depth(lines: List[str]) -> int:
    depth = 1
    for line in lines:
        if LIST_BULLET_RE.match(line.strip()):
            indent = len(line) - len(line.lstrip(" "))
            depth = max(depth, indent // 2 + 1)
    return depth


def classify_region(text: str, has_page_images: bool, figure_hint: bool = False) -> Tuple[str, Dict[str, object]]:
    lines = [line for line in text.splitlines() if line.strip()]
    first_line = lines[0].strip() if lines else ""
    header_level = detect_header_level(first_line)
    metadata: Dict[str, object] = {"header_level": header_level, "formula_detected": False}
    trimmed = text.strip()
    if figure_hint or (has_page_images and not trimmed):
        metadata["region_type"] = "figure"
        return "figure", metadata
    if any(TABLE_BORDER_RE.search(line) for line in lines):
        metadata["table_signature"] = extract_table_signature(lines)
        metadata["region_type"] = "table"
        return "table", metadata
    list_hits = sum(1 for line in lines if LIST_BULLET_RE.match(line.strip()))
    if list_hits >= max(1, len(lines) // 2):
        metadata["list_depth"] = estimate_list_depth(lines)
        metadata["region_type"] = "list"
        return "list", metadata
    if FORMULA_RE.search(text):
        metadata["formula_detected"] = True
        metadata["region_type"] = "formula"
        return "formula", metadata
    metadata["region_type"] = "text" if header_level == 0 else "heading"
    return ("text" if header_level == 0 else "heading"), metadata


def detect_paragraph_region(paragraph: str) -> str:
    para = paragraph.strip()
    if not para:
        return "text"
    lowered = para.lower()
    if lowered.startswith("figure") or lowered.startswith("fig "):
        return "figure"
    if any(TABLE_BORDER_RE.search(line) for line in para.splitlines()):
        return "table"
    if LIST_BULLET_RE.match(para):
        return "list"
    if FORMULA_RE.search(para):
        return "formula"
    if detect_header_level(para.splitlines()[0]) > 0:
        return "heading"
    return "text"


def noise_metrics(text: str) -> Dict[str, float]:
    total_chars = len(text) or 1
    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    digits = sum(1 for ch in text if ch.isdigit())
    symbols = sum(1 for ch in text if ch in {"@", "#", "$", "%", "^", "&"})
    whitespace_runs = sum(1 for part in text.split() if len(part) == 1 and not part.isalpha())
    newline_count = text.count("\n") or 1
    avg_line_len = total_chars / newline_count
    broken_words = text.count("-\n")
    tokens = WORD_RE.findall(text)
    unique_ratio = len(set(tokens)) / len(tokens) if tokens else 0.0
    score = (
        0.4 * (non_ascii / total_chars)
        + 0.2 * (broken_words / newline_count)
        + 0.2 * (symbols / total_chars)
        + 0.2 * (1 - unique_ratio)
    )
    return {
        "non_ascii_ratio": non_ascii / total_chars,
        "digit_ratio": digits / total_chars,
        "symbol_ratio": symbols / total_chars,
        "avg_line_length": avg_line_len,
        "broken_words": broken_words,
        "unique_token_ratio": unique_ratio,
        "noise_score": min(1.0, score),
        "format_split_errors": whitespace_runs / max(1, len(tokens)),
    }


def ocr_consensus_score(transcripts: Sequence[str]) -> float:
    normalized = []
    for text in transcripts:
        candidate = " ".join(text.split()).strip().lower()
        if candidate:
            normalized.append(candidate)
    if not normalized:
        return 0.0
    counts: Counter[str] = Counter(normalized)
    return max(counts.values()) / len(normalized)


def aggressive_math_cleanup(text: str) -> str:
    replacements = {
        "−": "-",
        "–": "-",
        "—": "-",
        "∗": "*",
        "ﬁ": "fi",
        "ﬂ": "fl",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = DIGIT_GAP_RE.sub("", text)
    text = re.sub(r"([=+\-])\s{2,}", r"\1 ", text)
    text = re.sub(r"\s{2,}([=+\-])", r" \1", text)
    return text


def chunk_text(
    pages: Sequence[str],
    page_images: Dict[int, List[str]],
    chunk_chars: int,
    ocr_helper: OCRFallback | None,
    page_store: PageImageStore | None,
    semantic_chunker: SemanticChunker | None,
    llm_sectioner: LLMSectioner | None = None,
    ambiguity_resolver: AmbiguityResolver | None = None,
    layout_segments: Dict[int, List[LayoutRegion]] | None = None,
    caption_verifier: ClipCaptionVerifier | None = None,
    math_classifier: MathContentClassifier | None = None,
    quality_mode: str = "balanced",
    layout_confidence_threshold: float | None = None,
) -> Tuple[List[common.Chunk], Dict[str, int], Dict[str, Dict[str, float]]]:
    chunks: List[common.Chunk] = []
    ocr_usage: Counter[str] = Counter()
    region_stats: Dict[str, Dict[str, float]] = defaultdict(lambda: {"count": 0, "avg_noise": 0.0})
    figure_fingerprints: Set[Tuple[str, ...]] = set()
    prev_embedding = None
    buffer_sentence_count = 0
    segments_map = layout_segments or {}
    classifier = math_classifier or MathContentClassifier()
    aggressive_mode = quality_mode == "aggressive"

    def _tag_chunk_math(chunk: common.Chunk) -> None:
        try:
            result = classifier.classify(chunk.text)
        except Exception:
            return
        chunk.metadata["math_role"] = result.label
        chunk.metadata["math_role_score"] = round(result.score, 3)
    layout_threshold = (
        layout_confidence_threshold
        if layout_confidence_threshold is not None
        else _layout_conf_threshold()
    )

    for page_idx, raw_text in enumerate(pages):
        text_sources: OrderedDict[str, str] = OrderedDict()
        base_text = raw_text.strip()
        if base_text:
            text_sources["pypdf"] = base_text
        if ocr_helper is not None:
            ocr_result = ocr_helper.extract(page_idx)
            for backend, transcript in ocr_result.sources.items():
                if backend not in text_sources:
                    text_sources[backend] = transcript
        page_image_path = None
        if page_store is not None:
            page_image = page_store.get_page_image(page_idx)
            if page_image:
                page_image_path = str(page_image)
        if caption_verifier:
            text_sources = caption_verifier.rank_sources(page_image_path, text_sources)
        page_consensus = 0.0
        if not text_sources:
            text = f"[ocr-missing page={page_idx + 1}]"
            backends_used = ["none"]
        else:
            merged = merge_text_sources(list(text_sources.values()))
            text = merged or "\n\n".join(text_sources.values())
            backends_used = list(text_sources.keys())
            page_consensus = ocr_consensus_score(text_sources.values())
        for backend in backends_used:
            ocr_usage[backend] += 1
        base_paragraphs = [para for para in text.split("\n\n") if para.strip()]
        if not base_paragraphs:
            base_paragraphs = [text]
        structured_records: List[Tuple[str, str | None, Dict[str, object]]] = []
        seen_normalized: Set[str] = set()
        for region in sorted(segments_map.get(page_idx, []), key=lambda seg: seg.order):
            region_conf = float((region.extras or {}).get("layout_confidence", 1.0) or 1.0)
            if region_conf < layout_threshold:
                continue
            normalized = " ".join(region.text.split())
            if not normalized:
                continue
            seen_normalized.add(normalized)
            meta = dict(region.extras)
            meta["tag"] = region.tag
            structured_records.append((region.text, region.tag, meta))
        for para in base_paragraphs:
            normalized = " ".join(para.split())
            if normalized and normalized in seen_normalized:
                continue
            structured_records.append((para, None, {}))
        if not structured_records:
            structured_records = [(text, None, {})]
        buffer: List[str] = []
        buffer_metadata: Dict[str, object] = {}
        buffer_region = None
        current_len = 0
        chunk_idx = 0
        llm_flush_indices: Set[int] = set()
        paragraphs_for_llm = [record[0] for record in structured_records]
        if llm_sectioner and paragraphs_for_llm:
            try:
                groups = llm_sectioner.plan(paragraphs_for_llm, chunk_chars)
            except Exception as exc:
                LOGGER.debug("LLM sectioner failed on page %s: %s", page_idx + 1, exc)
                groups = []
            for group in groups:
                if group:
                    llm_flush_indices.add(group[-1])
        paragraph_idx = 0
        def emit_chunk() -> None:
            nonlocal buffer, buffer_metadata, buffer_region, current_len, chunk_idx, buffer_sentence_count, figure_fingerprints
            if not buffer:
                return
            combined = "\n\n".join(buffer)
            layout_meta = buffer_metadata.copy() if buffer_metadata else None
            chunk = _build_chunk(
                page_idx,
                chunk_idx,
                combined,
                page_images,
                backends_used,
                page_image_path,
                buffer_region,
                layout_meta,
                page_consensus,
            )
            _tag_chunk_math(chunk)
            if aggressive_mode and chunk.metadata.get("region_type") in {"formula", "heading", "text"}:
                cleaned = aggressive_math_cleanup(chunk.text)
                if cleaned != chunk.text:
                    chunk.text = cleaned
                    chunk.metadata["aggressive_cleanup"] = True
            if ambiguity_resolver:
                try:
                    ambiguity_resolver.maybe_fix(chunk)
                except Exception:
                    pass
            if page_consensus < 0.6 and len(backends_used) > 1:
                chunk.metadata["ocr_review_required"] = True
            appended = True
            if chunk.metadata.get("region_type") == "figure" and chunk.images:
                fingerprint = tuple(sorted(chunk.images))
                if fingerprint in figure_fingerprints:
                    LOGGER.debug("Skipping duplicate figure assets on page %s", page_idx + 1)
                    appended = False
                else:
                    figure_fingerprints.add(fingerprint)
            if appended:
                chunks.append(chunk)
                noise = chunk.metadata.get("noise_score", 0.0)
                stats = region_stats[chunk.metadata.get("region_type", buffer_region or "text")]
                stats["avg_noise"] = (stats["avg_noise"] * stats["count"] + noise) / (stats["count"] + 1)
                stats["count"] += 1
            buffer = []
            buffer_metadata = {}
            current_len = 0
            chunk_idx += 1
            buffer_sentence_count = 0
            buffer_region = None

        for para, hint_tag, extra_meta in structured_records:
            para = para.strip()
            if not para:
                continue
            region = hint_tag or detect_paragraph_region(para)
            para_len = len(para)
            flush = False
            embedding = semantic_chunker.embed(para) if semantic_chunker else None
            sentence_count = semantic_chunker.sentence_count(para) if semantic_chunker else max(1, para.count("."))
            if not buffer_region:
                buffer_region = region
            if region != buffer_region:
                flush = True
            if region in {"heading", "question", "answer", "figure", "table", "formula"} and buffer:
                flush = True
            if current_len + para_len > chunk_chars and buffer:
                flush = True
            semantic_break = (
                semantic_chunker.should_break(prev_embedding, embedding, buffer_sentence_count)
                if semantic_chunker and buffer
                else False
            )
            if semantic_break:
                flush = True
            paragraph_idx += 1
            if flush and buffer:
                emit_chunk()
            buffer.append(para)
            current_len += para_len
            if extra_meta:
                for key, value in extra_meta.items():
                    if value is None or key == "text":
                        continue
                    buffer_metadata.setdefault(key, value)
            if hint_tag:
                buffer_region = hint_tag
                buffer_metadata.setdefault("tag", hint_tag)
            else:
                buffer_region = region
            buffer_sentence_count += sentence_count or 1
            if embedding is not None:
                prev_embedding = embedding
            if paragraph_idx in llm_flush_indices and buffer:
                emit_chunk()
                llm_flush_indices.discard(paragraph_idx)
        if buffer:
            emit_chunk()
    for idx, chunk in enumerate(chunks):
        prev_region = chunks[idx - 1].metadata.get("region_type") if idx > 0 else None
        next_region = chunks[idx + 1].metadata.get("region_type") if idx + 1 < len(chunks) else None
        chunk.metadata["context"] = {
            "prev_region": prev_region,
            "next_region": next_region,
            "prev_chunk_id": chunks[idx - 1].chunk_id if idx > 0 else None,
            "next_chunk_id": chunks[idx + 1].chunk_id if idx + 1 < len(chunks) else None,
        }
    region_stats = {region: stats for region, stats in region_stats.items()}
    return chunks, dict(ocr_usage), region_stats


def _build_chunk(
    page_idx: int,
    chunk_idx: int,
    text: str,
    page_images: Dict[int, List[str]],
    ocr_backends: List[str],
    page_image_path: str | None,
    region_hint: str | None,
    layout_metadata: Dict[str, object] | None = None,
    ocr_consensus: float = 1.0,
) -> common.Chunk:
    page_assets = page_images.get(page_idx, [])
    region_type, region_metadata = classify_region(
        text,
        bool(page_assets),
        figure_hint=(region_hint == "figure"),
    )
    if region_hint and region_type == "text":
        region_type = region_hint
        region_metadata["region_type"] = region_hint
    if layout_metadata:
        hint = layout_metadata.get("tag")
        if hint:
            region_type = hint
            region_metadata["region_type"] = hint
    primary_backend = ocr_backends[0] if ocr_backends else "none"
    metadata: Dict[str, object] = {
        "ocr_backend": primary_backend,
        "ocr_backends": ocr_backends,
        "ocr_consensus": round(float(ocr_consensus), 3),
        "ocr_multi_pass": len(ocr_backends),
        "paragraphs": text.count("\n") + 1,
        "page_image": page_image_path,
        "region_type": region_type,
    }
    metadata.update(region_metadata)
    if layout_metadata:
        for key, value in layout_metadata.items():
            if key == "text":
                continue
            metadata.setdefault(key, value)
    if "layout_confidence" in metadata and metadata["layout_confidence"] is not None:
        try:
            metadata["layout_confidence"] = round(float(metadata["layout_confidence"]), 3)
        except (TypeError, ValueError):
            pass
    metadata["image_refs"] = page_assets if region_type == "figure" else []
    metadata.update(noise_metrics(text))
    branch_id = None
    if layout_metadata:
        branch_id = layout_metadata.get("branch_id")
    branch_provenance: Dict[str, object] = {"primary": "ocr"}
    if branch_id:
        branch_provenance["vision"] = {"branch_id": branch_id}
        metadata["vision_branch_id"] = branch_id
    metadata["branch_provenance"] = branch_provenance
    return common.Chunk(
        chunk_id=f"page{page_idx + 1:03d}_{chunk_idx:02d}",
        page=page_idx + 1,
        text=text,
        images=page_assets if region_type == "figure" else [],
        metadata=metadata,
    )


def _enrich_chunks_with_vision_branches(
    chunks: Sequence[common.Chunk],
    branches: Dict[str, VisionSynthesisBranch],
) -> float:
    if not branches:
        return 0.0
    resolved = 0
    total = len(chunks)
    for chunk in chunks:
        metadata = chunk.metadata or {}
        provenance = dict(metadata.get("branch_provenance") or {})
        branch_id = None
        vision_meta = provenance.get("vision")
        if isinstance(vision_meta, dict):
            branch_id = vision_meta.get("branch_id")
        if not branch_id:
            branch_id = metadata.get("vision_branch_id")
        if not branch_id:
            continue
        branch = branches.get(branch_id)
        if not branch:
            continue
        provenance["vision"] = branch.to_metadata()
        provenance.setdefault("primary", "ocr")
        metadata["branch_provenance"] = provenance
        resolved += 1
    return resolved / max(1, total)


def run_ingestion(
    pdf_path: Path,
    workspace: Path,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    ocr_mode: str = "auto",
    capture_page_images: bool = False,
    models_dir: Path | None = None,
    semantic_chunker: SemanticChunker | None = None,
    telemetry: TelemetryCallback | None = None,
    backend_config: BackendToggleConfig | None = None,
    vision_branch_enabled: bool | None = None,
    layout_confidence_threshold: float | None = None,
    enable_monkey_ocr: bool | None = None,
) -> IngestionResult:
    resolved_mode = (
        backend_config.resolve_ingestion_mode() if backend_config else ocr_mode
    )
    backend_label = backend_config.ocr_backend if backend_config else ocr_mode
    math_backend = backend_config.math_ocr_backend if backend_config else "none"
    mode = resolved_mode.lower()
    if mode not in OCR_MODES:
        raise ValueError(f"Unsupported OCR mode: {ocr_mode}")
    models_path = (models_dir or MODELS_ROOT).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    image_dir = workspace / "images"
    ocr_dir = workspace / "ocr"
    page_cache_dir = workspace / "page_rasters"
    pdf_path = pdf_path.resolve()
    LOGGER.info("Reading PDF %s", pdf_path)
    pages = read_pdf_text(pdf_path)
    quality_analyzer = InputQualityAssessor()
    math_page_hints = [bool(FORMULA_RE.search(text)) for text in pages]
    images = run_pdfimages(pdf_path, image_dir)
    preview_quality = quality_analyzer.preview_from_pages(pages, images)
    mapping = assign_images_to_pages(images, len(pages))
    render_modes = {"auto", "nougat", "pytesseract", "florence2", "internvl", "mathvision", "mathocr"}
    layout_backend = (
        (backend_config.layout_backend if backend_config else None)
        or os.environ.get("LATEXIFY_LAYOUT_BACKEND")
        or "pymupdf"
    ).lower()
    need_renders = capture_page_images or mode in render_modes or layout_backend == "surya"
    page_store = PageImageStore(pdf_path, page_cache_dir, enabled=need_renders)
    threshold = (
        layout_confidence_threshold if layout_confidence_threshold is not None else _layout_conf_threshold()
    )
    monkey_adapter = None
    use_monkey_ocr = (
        enable_monkey_ocr
        if enable_monkey_ocr is not None
        else os.environ.get("LATEXIFY_ENABLE_MONKEY_OCR", "1").lower() not in {"0", "false", "off"}
    )
    if use_monkey_ocr:
        try:
            monkey_adapter = MonkeyOCRAdapter(pdf_path)
            if not monkey_adapter.available():
                monkey_adapter = None
        except Exception as exc:  # pragma: no cover - defensive guard
            LOGGER.warning("MonkeyOCR initialization failed: %s", exc)
            monkey_adapter = None
    if vision_branch_enabled is None:
        vision_branch_enabled = os.environ.get("LATEXIFY_ENABLE_VISION_SYNTHESIS", "1") != "0"
    structure_analyzer = DocumentStructureAnalyzer(
        pdf_path,
        enable_layoutlm=True,
        enable_vision_branch=vision_branch_enabled,
        monkey_adapter=monkey_adapter,
        layout_backend=layout_backend,
        page_image_store=page_store,
    )
    layout_segments, document_tree = structure_analyzer.analyze_document(len(pages))
    master_ocr_items_path = _export_master_ocr_items(layout_segments, workspace / "master_ocr_items.json")
    clip_verifier = ClipCaptionVerifier()
    ocr_helper = (
        None
        if mode == "none"
        else OCRFallback(
            mode,
            ocr_dir,
            page_store,
            models_path,
            telemetry=telemetry,
            math_page_hints=math_page_hints,
            quality_profile=preview_quality,
            monkey_adapter=monkey_adapter,
        )
    )
    chunker = semantic_chunker or SemanticChunker()
    llm_sectioner = None
    if os.environ.get("LATEXIFY_ENABLE_LLM_SECTIONING", "1") != "0":
        llm_sectioner = build_sectioner()
    ambiguity_resolver = None
    if os.environ.get("LATEXIFY_ENABLE_VLM_AMBIGUITY", "1") != "0":
        try:
            ambiguity_resolver = AmbiguityResolver()
        except Exception as exc:
            LOGGER.info("Ambiguity resolver unavailable: %s", exc)
            ambiguity_resolver = None
    quality_mode = preview_quality.processing_mode if preview_quality else "balanced"
    if ocr_helper and getattr(ocr_helper, "_sequential_mode", False):
        try:
            ocr_helper.prepare_document(len(pages))
        except Exception as exc:
            LOGGER.warning("Sequential OCR preparation failed: %s", exc)
    branch_future = None
    chunks: List[common.Chunk] = []
    ocr_usage: Dict[str, int] = {}
    region_stats: Dict[str, Dict[str, float]] = {}
    vision_branch_map: Dict[str, VisionSynthesisBranch] = {}
    worker_count = 2 if vision_branch_enabled else 1
    try:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            chunk_future = executor.submit(
                chunk_text,
                pages=pages,
                page_images=mapping,
                chunk_chars=chunk_chars,
                ocr_helper=ocr_helper,
                page_store=page_store if need_renders else None,
                semantic_chunker=chunker,
                llm_sectioner=llm_sectioner,
                ambiguity_resolver=ambiguity_resolver,
                layout_segments=layout_segments,
                caption_verifier=clip_verifier if clip_verifier.available() else None,
                quality_mode=quality_mode,
                layout_confidence_threshold=threshold,
            )
            if vision_branch_enabled:
                branch_future = executor.submit(
                    structure_analyzer.build_vision_branches,
                    layout_segments,
                    page_store if need_renders else None,
                )
            chunks, ocr_usage, region_stats = chunk_future.result()
            if branch_future:
                vision_branch_map = branch_future.result()
    finally:
        if ambiguity_resolver:
            try:
                ambiguity_resolver.close()
            except Exception:
                LOGGER.debug("Ambiguity resolver shutdown failed.", exc_info=True)
    math_role_counts = Counter(
        chunk.metadata.get("math_role", "unknown") for chunk in chunks
    )
    vision_branch_ratio = _enrich_chunks_with_vision_branches(chunks, vision_branch_map)
    _dedupe_repetitive_lines(chunks, len(pages))
    quality_profile = quality_analyzer.summarize_chunks(
        chunks,
        page_cache_dir if page_store.enabled else None,
        preview_quality,
    )
    structure_analyzer.close()
    if ocr_helper:
        ocr_helper.shutdown()
    if monkey_adapter:
        monkey_adapter.close()
    chunks_path = workspace / "chunks.json"
    common.save_chunks(chunks, chunks_path)
    document = _build_document_representation(
        chunks,
        len(pages),
        backend_label,
        math_backend,
        mode,
    )
    document_path = workspace / "document.json"
    common.save_document(document, document_path)
    try:
        document_rel_path = str(document_path.relative_to(workspace))
    except Exception:
        document_rel_path = str(document_path)
    total_blocks = sum(len(page.blocks) for page in document.pages)
    tree_path = workspace / "document_tree.json"
    tree_payload = {
        f"page_{page_idx + 1:04d}": nodes for page_idx, nodes in (document_tree or {}).items()
    }
    tree_path.write_text(json.dumps(tree_payload, indent=2), encoding="utf-8")
    LOGGER.info("Document structure written to %s", tree_path)
    manifest = workspace / "ingestion_manifest.json"
    backend_meta = backend_config.as_dict() if backend_config else {
        "ocr_backend": backend_label,
        "mineru_enabled": False,
        "marker_enabled": False,
        "mcp_pdf_processor_enabled": False,
        "math_ocr_backend": "none",
    }
    backend_meta["resolved_ocr_mode"] = mode
    backend_meta["requested_backend"] = backend_label
    manifest_payload = {
        "pdf": str(pdf_path),
        "pages": len(pages),
        "images": len(images),
        "chunk_chars": chunk_chars,
        "chunks": len(chunks),
        "ocr_mode": mode,
        "ocr_backends": {
            "pytesseract": pytesseract is not None,
            "nougat": (models_path / "ocr" / "nougat-small").exists(),
            "florence2": (models_path / "ocr" / "florence-2-large").exists(),
            "internvl": (models_path / INTERNVL_MODEL_SUBPATH).exists(),
            "mathvision": (models_path / "ocr" / "trocr-math").exists(),
            "mathocr": (models_path / "ocr" / "pix2tex-base").exists(),
            "monkeyocr": bool(monkey_adapter and monkey_adapter.available()),
        },
        "page_images_available": bool(page_store.enabled),
        "ocr_usage": ocr_usage,
        "region_noise_metrics": region_stats,
        "layout_analysis": {
            "enabled": bool(layout_segments),
            "segments": sum(len(regions) for regions in layout_segments.values()),
        },
        "clip_verifier": clip_verifier.available(),
        "document_tree": {
            "path": str(tree_path.relative_to(workspace)),
            "pages": len(document_tree),
            "nodes": sum(len(nodes) for nodes in (document_tree or {}).values()),
        },
        "master_ocr_items": {
            "path": str(master_ocr_items_path.relative_to(workspace)) if master_ocr_items_path else None,
            "regions": sum(len(regions) for regions in layout_segments.values()) if layout_segments else 0,
        },
        "vision_branch": {
            "enabled": vision_branch_enabled,
            "branches": len(vision_branch_map),
            "chunk_coverage": round(vision_branch_ratio, 3),
        },
        "document_representation": {
            "path": document_rel_path,
            "pages": len(document.pages),
            "blocks": total_blocks,
            "source_backend": document.source_backend,
            "math_ocr_backend": math_backend,
        },
        "math_roles": dict(math_role_counts),
        "input_quality": quality_profile.to_dict(),
        "backend_config": backend_meta,
    }
    manifest_payload["ocr_available"] = any(manifest_payload["ocr_backends"].values())
    manifest.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    LOGGER.info("Ingestion complete with %s chunks", len(chunks))
    page_dir = page_cache_dir if page_store.enabled else None
    return IngestionResult(
        chunks_path=chunks_path,
        image_dir=image_dir,
        ocr_dir=ocr_dir,
        document_path=document_path,
        page_images_dir=page_dir,
        page_images_available=page_store.enabled,
        tree_path=tree_path,
        manifest_path=manifest,
        quality_profile=quality_profile.to_dict(),
        vision_branch_summary=manifest_payload.get("vision_branch"),
        master_ocr_items_path=master_ocr_items_path,
    )


def _canonical_block_type(region: str | None) -> str:
    if not region:
        return "text"
    normalized = region.lower()
    mapping = {
        "formula": "equation",
        "equation": "equation",
        "math": "equation",
        "table": "table",
        "tabular": "table",
        "figure": "figure",
        "image": "figure",
        "heading": "heading",
        "title": "heading",
        "list": "list",
        "question": "question",
        "answer": "answer",
    }
    return mapping.get(normalized, "text")


def _build_document_representation(
    chunks: Sequence[common.Chunk],
    total_pages: int,
    backend_label: str,
    math_backend: str,
    resolved_mode: str,
) -> common.Document:
    max_page = max((chunk.page for chunk in chunks), default=0)
    page_total = max(total_pages, max_page)
    if page_total <= 0:
        page_total = 1
    pages = [
        common.DocumentPage(page_number=idx + 1, source_backend=backend_label)
        for idx in range(page_total)
    ]
    active_backends: Set[str] = set()
    for chunk in chunks:
        block_type = _canonical_block_type(chunk.metadata.get("region_type"))
        block_backend = str(chunk.metadata.get("ocr_backend", backend_label))
        active_backends.add(block_backend)
        consensus_raw = chunk.metadata.get("ocr_consensus")
        try:
            consensus = float(consensus_raw) if consensus_raw is not None else None
        except (TypeError, ValueError):
            consensus = None
        span_metadata = {}
        backends_used = chunk.metadata.get("ocr_backends")
        if backends_used:
            span_metadata["ocr_backends"] = backends_used
        if consensus is not None:
            span_metadata["ocr_consensus"] = consensus
        span = common.DocumentSpan(
            text=chunk.text,
            source_backend=block_backend,
            confidence=consensus,
            metadata=span_metadata,
        )
        block = common.DocumentBlock(
            block_id=chunk.chunk_id,
            page_number=chunk.page,
            block_type=block_type,
            text=chunk.text,
            spans=[span],
            source_backend=block_backend,
            images=list(chunk.images),
            metadata=dict(chunk.metadata),
        )
        if block_type == "equation":
            block.equation = common.DocumentEquation(
                latex=None,
                raw_text=chunk.text,
                confidence=chunk.metadata.get("math_role_score"),
                metadata={
                    "math_role": chunk.metadata.get("math_role"),
                    "math_ocr_backend": math_backend,
                },
            )
        if block_type == "table":
            block.table = common.DocumentTable(
                rows=int(chunk.metadata.get("table_rows", 0) or 0),
                cols=int(chunk.metadata.get("table_cols", 0) or 0),
                metadata={
                    "signature": chunk.metadata.get("table_signature"),
                },
            )
        page_index = min(max(chunk.page, 1), page_total) - 1
        pages[page_index].blocks.append(block)
    doc_metadata = {
        "math_ocr_backend": math_backend,
        "available_ocr_backends": sorted(active_backends) if active_backends else [backend_label],
        "resolved_ocr_mode": resolved_mode,
    }
    return common.Document(
        pages=pages,
        source_backend=backend_label,
        metadata=doc_metadata,
    )


__all__ = [
    "run_ingestion",
    "IngestionResult",
    "OCR_MODES",
    "merge_text_sources",
    "OCRResult",
    "DocumentStructureAnalyzer",
    "INTERNVL_MODEL_ID",
    "INTERNVL_MODEL_SUBPATH",
]


def _dedupe_repetitive_lines(chunks: List[common.Chunk], total_pages: int) -> None:
    """Remove repeated headers/footers across pages before planning."""

    if total_pages <= 1:
        return
    header_counts: Counter[str] = Counter()
    footer_counts: Counter[str] = Counter()
    page_first: Dict[int, str] = {}
    page_last: Dict[int, str] = {}
    for chunk in chunks:
        metadata = chunk.metadata or {}
        page = metadata.get("page")
        if page is None:
            continue
        lines = [line.strip() for line in chunk.text.splitlines() if line.strip()]
        if not lines:
            continue
        page_first.setdefault(page, lines[0])
        page_last[page] = lines[-1]
    for line in page_first.values():
        if line:
            header_counts[line] += 1
    for line in page_last.values():
        if line:
            footer_counts[line] += 1
    min_occurrences = max(3, total_pages // 3)
    header_set = {line for line, count in header_counts.items() if count >= min_occurrences}
    footer_set = {line for line, count in footer_counts.items() if count >= min_occurrences}
    if not header_set and not footer_set:
        return
    for chunk in chunks:
        lines = chunk.text.splitlines()
        start = 0
        end = len(lines)
        while start < end and lines[start].strip() in header_set:
            start += 1
        while end > start and lines[end - 1].strip() in footer_set:
            end -= 1
        if start == 0 and end == len(lines):
            continue
        chunk.text = "\n".join(lines[start:end]).strip()
def merge_text_sources(sources: Sequence[str]) -> str:
    """Merge multiple OCR transcripts by deduplicating paragraphs."""

    merged: List[str] = []
    seen = set()
    for text in sources:
        for paragraph in (seg.strip() for seg in text.split("\n\n")):
            if not paragraph:
                continue
            normalized = " ".join(paragraph.split())
            if normalized in seen:
                continue
            seen.add(normalized)
            merged.append(paragraph)
    return "\n\n".join(merged)
