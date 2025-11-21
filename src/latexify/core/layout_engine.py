import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import re

try:
    import fitz  # type: ignore
except ImportError:
    fitz = None

LOGGER = logging.getLogger(__name__)

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
            except Exception as exc:
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
            except Exception as exc:
                LOGGER.debug("LayoutAnalyzer skipped page %s: %s", page_idx + 1, exc)
                continue
        return regions

    def _analyze_page(self, page_index: int) -> List[LayoutRegion]:
        doc = self._load_doc()
        if doc is None:
            return []
        try:
            page = doc.load_page(page_index)
        except Exception:
            return []
            
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
        
        # Basic Regex Constants (Simplification of ingestion.py regexes)
        QUESTION_RE = re.compile(r"^(question|q)\\s*([0-9]+[a-z]?|\\([^)]+\\))", re.IGNORECASE)
        ANSWER_RE = re.compile(r"^(answer|solution)\\b", re.IGNORECASE)
        TABLE_BORDER_RE = re.compile(r"(\\+[-+]+\\+)|(\\|.+\\||)")
        LIST_BULLET_RE = re.compile(r"^([0-9]+\\.[\\)\\s]|[A-Za-z]\\.|[-*•])\\s+")
        FORMULA_RE = re.compile(r"(\\begin{document}{equation}|\\frac|\\sum|\\int|=|\[a-z]+)")
        
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
            return "list", extras
        if font_size >= 18: 
            return "heading", extras
        if FORMULA_RE.search(text):
            extras["formula_detected"] = True
            return "formula", extras
        return "text", extras
