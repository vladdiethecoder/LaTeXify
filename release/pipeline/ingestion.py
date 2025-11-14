"""Ingestion stage for the self-contained LaTeXify release pipeline."""
from __future__ import annotations

import importlib
import json
import logging
import math
import os
import re
import subprocess
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, Set

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


from ..core import common
from ..utils.ensemble import EnsembleVoter
from .semantic_chunking import SemanticChunker
from .math_classifier import MathContentClassifier

LOGGER = logging.getLogger(__name__)
DEFAULT_CHUNK_CHARS = 1200
MODELS_ROOT = Path(__file__).resolve().parents[1] / "models"
OCR_MODES = {"auto", "pytesseract", "nougat", "florence2", "internvl", "mathvision", "mathocr", "none"}
OCR_MODEL_SPECS = {
    "nougat": {"repo_id": "facebook/nougat-small", "subpath": Path("ocr") / "nougat-small"},
    "florence2": {"repo_id": "microsoft/Florence-2-large-ft", "subpath": Path("ocr") / "florence-2-large"},
    "internvl": {"repo_id": "OpenGVLab/InternVL-Chat-V1-2", "subpath": Path("ocr") / "internvl-3.5-14b"},
    "mathvision": {"repo_id": "microsoft/trocr-base-handwritten", "subpath": Path("ocr") / "trocr-math"},
    "mathocr": {"repo_id": "lupantech/pix2tex-base", "subpath": Path("ocr") / "pix2tex-base"},
}
FORCE_HEAVY_OCR = os.environ.get("LATEXIFY_OCR_FORCE_HEAVY", "0") == "1"
GPU_PREF_ENV = os.environ.get("LATEXIFY_OCR_GPU_PREF")
RELEASE_HEAVY_MODE = os.environ.get("LATEXIFY_OCR_RELEASE_MODE", "run")
BACKEND_PYTHON_DEPS = {
    "florence2": ("einops", "timm"),
    "internvl": ("einops",),
    "mathocr": ("pix2tex", "ultralytics"),
}
FORMULA_RE = re.compile(r"(\\begin\{equation\}|\\frac|\\sum|\\int|=|\\[a-z]+)")
TABLE_BORDER_RE = re.compile(r"(\+[-+]+\+)|(\|.+\|)")
LIST_BULLET_RE = re.compile(r"^([0-9]+\.[\)\s]|[A-Za-z]\.|[-*•])\s+")
HEADER_KEYWORDS = ("chapter", "section", "appendix", "part", "lesson")
WORD_RE = re.compile(r"[A-Za-z]+")
QUESTION_RE = re.compile(r"^(question|q)\s*([0-9]+[a-z]?|\([^)]+\))", re.IGNORECASE)
ANSWER_RE = re.compile(r"^(answer|solution)\b", re.IGNORECASE)


@dataclass
class IngestionResult:
    chunks_path: Path
    image_dir: Path
    ocr_dir: Path
    page_images_dir: Path | None = None
    page_images_available: bool = False


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
            })
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


class DocumentStructureAnalyzer:
    """Higher-level structural analyzer that emits per-page document trees."""

    def __init__(
        self,
        pdf_path: Path,
        enable_layoutlm: bool = True,
        model_name: str = "microsoft/layoutlmv3-base",
        label_map: Dict[int, str] | None = None,
    ) -> None:
        self._layout = LayoutAnalyzer(pdf_path, enabled=True)
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
        if self._use_layoutlm:
            try:  # pragma: no cover - heavy dependency
                self._processor = LayoutLMv3Processor.from_pretrained(model_name)
                self._model = LayoutLMv3ForTokenClassification.from_pretrained(model_name)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    def close(self) -> None:
        self._layout.close()


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

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
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

    @property
    def backends(self) -> List[str]:
        return list(self.sources.keys())


class OCRFallback:
    """OCR helper that can route between multiple local models."""

    def __init__(
        self,
        mode: str,
        cache_dir: Path,
        page_store: PageImageStore | None,
        models_dir: Path | None,
    ) -> None:
        self.mode = mode
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.page_store = page_store
        self.models_dir = models_dir or MODELS_ROOT
        self._nougat = None
        self._florence = None
        self._internvl = None
        self._mathvision = None
        self._mathocr_engine = None
        self._device_overrides: Dict[str, str] = {}
        self._availability = {
            "nougat": True,
            "florence2": True,
            "internvl": True,
            "mathvision": True,
            "mathocr": MathOCREngine is not None,
            "pytesseract": pytesseract is not None,
        }
        self._download_attempted: Dict[str, bool] = {key: False for key in OCR_MODEL_SPECS}
        self._memory_requirements = {
            "florence2": 10 * 1024**3,  # ~10 GB
            "internvl": 22 * 1024**3,  # ~22 GB
            "mathvision": 4 * 1024**3,
        }
        self._gpu_reservations: Dict[int, int] = defaultdict(int)
        self._gpu_preference = self._parse_gpu_preference(GPU_PREF_ENV)
        self._persistent_backends = os.environ.get("LATEXIFY_OCR_KEEP_LIVE", "0") == "1"

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
        if torch is None:
            raise RuntimeError("PyTorch is required for OCR backends but is unavailable.")
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
        requirement = self._memory_requirements.get(backend)
        picked_idx: int | None = None
        for idx, total, free in inventory:
            if exclude and idx in exclude:
                continue
            reserved = self._gpu_reservations.get(idx, 0)
            available = free - reserved if not FORCE_HEAVY_OCR else total - reserved
            if requirement is None or FORCE_HEAVY_OCR:
                picked_idx = idx
                break
            if available >= requirement:
                picked_idx = idx
                break
        if picked_idx is None:
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

    def _release_backend(self, backend: str) -> None:
        released = False
        device = self._device_overrides.pop(backend, None)
        if backend == "florence2" and self._florence is not None:
            self._florence = None
            released = True
        if backend == "internvl" and self._internvl is not None:
            self._internvl = None
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
            from ..models.model_adapters import NougatAdapter, NougatConfig
        except Exception as exc:  # pragma: no cover - optional import
            LOGGER.warning("Failed to import Nougat adapter: %s", exc)
            if self._availability.get("nougat", True):
                LOGGER.warning("Failed to import Nougat adapter: %s", exc)
                self._availability["nougat"] = False
            return None
        try:
            self._nougat = NougatAdapter(NougatConfig(model_dir=model_dir))
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
            try:
                self._florence = FlorenceAdapter(FlorenceConfig(model_dir=model_dir, device=device))
                self._device_overrides["florence2"] = device
                self._reserve_gpu(device_idx, "florence2")
                LOGGER.info("florence2 backend assigned to %s", device)
                return self._florence
            except RuntimeError as exc:
                if "CUDA out of memory" in str(exc) and device_idx is not None:
                    exclude.add(device_idx)
                    continue
                LOGGER.warning("Failed to initialize Florence model: %s", exc)
                self._florence = None
                self._availability["florence2"] = False
                self._device_overrides.pop("florence2", None)
                return None

    def _lazy_internvl(self):
        if self._internvl is not None:
            return self._internvl
        if not self._availability.get("internvl", True):
            return None
        if not self._ensure_python_packages("internvl"):
            return None
        try:
            from ..models.model_adapters import InternVLAdapter, InternVLConfig
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
        exclude: set[int] = set()
        while True:
            device, device_idx = self._select_backend_device("internvl", exclude)
            try:
                self._internvl = InternVLAdapter(InternVLConfig(model_dir=model_dir, device=device))
                self._device_overrides["internvl"] = device
                self._reserve_gpu(device_idx, "internvl")
                LOGGER.info("internvl backend assigned to %s", device)
                return self._internvl
            except RuntimeError as exc:
                if "CUDA out of memory" in str(exc) and device_idx is not None:
                    exclude.add(device_idx)
                    continue
                LOGGER.warning("Failed to initialize InternVL: %s", exc)
                self._internvl = None
                self._availability["internvl"] = False
                self._device_overrides.pop("internvl", None)
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
            self._mathvision = MathVisionAdapter(MathVisionConfig(model_dir=model_dir))
            LOGGER.info("mathvision backend initialized for math-heavy OCR.")
            return self._mathvision
        except RuntimeError as exc:
            LOGGER.warning("Failed to initialize MathVision model: %s", exc)
            self._availability["mathvision"] = False
            self._mathvision = None
            return None

    def _page_image(self, page_index: int) -> Path | None:
        if self.page_store is None:
            return None
        return self.page_store.get_page_image(page_index)

    def _run_nougat(self, page_index: int) -> str | None:
        nougat = self._lazy_nougat()
        if nougat is None:
            return None
        image = self._page_image(page_index)
        if not image:
            return None
        try:
            return nougat.predict(image)
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
            if RELEASE_HEAVY_MODE == "page":
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
            if RELEASE_HEAVY_MODE == "page":
                self._release_backend("internvl")

    def _run_tesseract(self, page_index: int) -> str | None:
        if pytesseract is None:
            return None
        image = self._page_image(page_index)
        if not image:
            return None
        text = pytesseract.image_to_string(image)
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
            self._mathocr_engine = MathOCREngine()
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
        attempt_order: List[str]
        if self.mode == "auto":
            attempt_order = ["nougat", "mathvision", "mathocr", "florence2", "internvl", "pytesseract"]
        elif self.mode == "nougat":
            attempt_order = ["nougat"]
        elif self.mode == "pytesseract":
            attempt_order = ["pytesseract"]
        elif self.mode == "florence2":
            attempt_order = ["florence2"]
        elif self.mode == "internvl":
            attempt_order = ["internvl"]
        elif self.mode == "mathvision":
            attempt_order = ["mathvision"]
        elif self.mode == "mathocr":
            attempt_order = ["mathocr"]
        else:
            attempt_order = []
        sources: OrderedDict[str, str] = OrderedDict()
        try:
            for backend in attempt_order:
                if backend == "nougat":
                    text = self._run_nougat(page_index)
                elif backend == "florence2":
                    text = self._run_florence(page_index)
                elif backend == "internvl":
                    text = self._run_internvl(page_index)
                elif backend == "mathvision":
                    text = self._run_mathvision(page_index)
                elif backend == "mathocr":
                    text = self._run_mathocr(page_index)
                else:
                    text = self._run_tesseract(page_index)
                if text:
                    cleaned = text.strip()
                    if cleaned:
                        sources[backend] = cleaned
        finally:
            if RELEASE_HEAVY_MODE == "page" and not self._persistent_backends:
                self._release_backend("florence2")
                self._release_backend("internvl")
                self._release_backend("mathvision")
                self._release_backend("mathocr")
        return OCRResult(sources)


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


def chunk_text(
    pages: Sequence[str],
    page_images: Dict[int, List[str]],
    chunk_chars: int,
    ocr_helper: OCRFallback | None,
    page_store: PageImageStore | None,
    semantic_chunker: SemanticChunker | None,
    layout_segments: Dict[int, List[LayoutRegion]] | None = None,
    caption_verifier: ClipCaptionVerifier | None = None,
    math_classifier: MathContentClassifier | None = None,
) -> Tuple[List[common.Chunk], Dict[str, int], Dict[str, Dict[str, float]]]:
    chunks: List[common.Chunk] = []
    ocr_usage: Counter[str] = Counter()
    region_stats: Dict[str, Dict[str, float]] = defaultdict(lambda: {"count": 0, "avg_noise": 0.0})
    figure_fingerprints: Set[Tuple[str, ...]] = set()
    prev_embedding = None
    buffer_sentence_count = 0
    segments_map = layout_segments or {}
    classifier = math_classifier or MathContentClassifier()

    def _tag_chunk_math(chunk: common.Chunk) -> None:
        try:
            result = classifier.classify(chunk.text)
        except Exception:
            return
        chunk.metadata["math_role"] = result.label
        chunk.metadata["math_role_score"] = round(result.score, 3)
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
        if not text_sources:
            text = f"[ocr-missing page={page_idx + 1}]"
            backends_used = ["none"]
        else:
            merged = merge_text_sources(list(text_sources.values()))
            text = merged or "\n\n".join(text_sources.values())
            backends_used = list(text_sources.keys())
        for backend in backends_used:
            ocr_usage[backend] += 1
        base_paragraphs = [para for para in text.split("\n\n") if para.strip()]
        if not base_paragraphs:
            base_paragraphs = [text]
        structured_records: List[Tuple[str, str | None, Dict[str, object]]] = []
        seen_normalized: Set[str] = set()
        for region in sorted(segments_map.get(page_idx, []), key=lambda seg: seg.order):
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
            if flush and buffer:
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
                )
                _tag_chunk_math(chunk)
                if chunk.metadata.get("region_type") == "figure" and chunk.images:
                    fingerprint = tuple(sorted(chunk.images))
                    if fingerprint in figure_fingerprints:
                        LOGGER.debug("Skipping duplicate figure assets on page %s", page_idx + 1)
                        buffer = []
                        current_len = 0
                        chunk_idx += 1
                        buffer_sentence_count = 0
                        buffer_region = None
                        buffer_metadata = {}
                        continue
                    figure_fingerprints.add(fingerprint)
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
        if buffer:
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
            )
            _tag_chunk_math(chunk)
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
            buffer_sentence_count = 0
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
    metadata["image_refs"] = page_assets if region_type == "figure" else []
    metadata.update(noise_metrics(text))
    return common.Chunk(
        chunk_id=f"page{page_idx + 1:03d}_{chunk_idx:02d}",
        page=page_idx + 1,
        text=text,
        images=page_assets if region_type == "figure" else [],
        metadata=metadata,
    )


def run_ingestion(
    pdf_path: Path,
    workspace: Path,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    ocr_mode: str = "auto",
    capture_page_images: bool = False,
    models_dir: Path | None = None,
    semantic_chunker: SemanticChunker | None = None,
) -> IngestionResult:
    mode = ocr_mode.lower()
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
    images = run_pdfimages(pdf_path, image_dir)
    mapping = assign_images_to_pages(images, len(pages))
    render_modes = {"auto", "nougat", "pytesseract", "florence2", "internvl", "mathvision", "mathocr"}
    need_renders = capture_page_images or mode in render_modes
    page_store = PageImageStore(pdf_path, page_cache_dir, enabled=need_renders)
    structure_analyzer = DocumentStructureAnalyzer(pdf_path, enable_layoutlm=True)
    layout_segments, document_tree = structure_analyzer.analyze_document(len(pages))
    clip_verifier = ClipCaptionVerifier()
    ocr_helper = None if mode == "none" else OCRFallback(mode, ocr_dir, page_store, models_path)
    chunker = semantic_chunker or SemanticChunker()
    chunks, ocr_usage, region_stats = chunk_text(
        pages,
        mapping,
        chunk_chars,
        ocr_helper,
        page_store if need_renders else None,
        chunker,
        layout_segments=layout_segments,
        caption_verifier=clip_verifier if clip_verifier.available() else None,
    )
    math_role_counts = Counter(
        chunk.metadata.get("math_role", "unknown") for chunk in chunks
    )
    structure_analyzer.close()
    if ocr_helper:
        ocr_helper.shutdown()
    chunks_path = workspace / "chunks.json"
    common.save_chunks(chunks, chunks_path)
    tree_path = workspace / "document_tree.json"
    tree_payload = {
        f"page_{page_idx + 1:04d}": nodes for page_idx, nodes in (document_tree or {}).items()
    }
    tree_path.write_text(json.dumps(tree_payload, indent=2), encoding="utf-8")
    LOGGER.info("Document structure written to %s", tree_path)
    manifest = workspace / "ingestion_manifest.json"
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
            "internvl": (models_path / "ocr" / "internvl-3.5-14b").exists(),
            "mathvision": (models_path / "ocr" / "trocr-math").exists(),
            "mathocr": (models_path / "ocr" / "pix2tex-base").exists(),
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
        "math_roles": dict(math_role_counts),
    }
    manifest_payload["ocr_available"] = any(manifest_payload["ocr_backends"].values())
    manifest.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    LOGGER.info("Ingestion complete with %s chunks", len(chunks))
    page_dir = page_cache_dir if page_store.enabled else None
    return IngestionResult(
        chunks_path=chunks_path,
        image_dir=image_dir,
        ocr_dir=ocr_dir,
        page_images_dir=page_dir,
        page_images_available=page_store.enabled,
    )


__all__ = [
    "run_ingestion",
    "IngestionResult",
    "OCR_MODES",
    "merge_text_sources",
    "OCRResult",
    "DocumentStructureAnalyzer",
]
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
