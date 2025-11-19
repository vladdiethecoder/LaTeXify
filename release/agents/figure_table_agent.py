"""Computer-vision assisted figure and table agent."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np

try:  # pragma: no cover - OpenCV is optional in CI
    import cv2
except Exception:  # pragma: no cover - graceful degradation when cv2 missing
    cv2 = None  # type: ignore

from ..core import common
from ..core.hierarchical_schema import ReferenceIndex

if TYPE_CHECKING:  # pragma: no cover - type checking helper
    from ..models.llm_refiner import LLMRefiner

LOGGER = logging.getLogger(__name__)


@dataclass
class PanelBox:
    """Bounding-box metadata for a detected subfigure."""

    x: int
    y: int
    w: int
    h: int
    area_ratio: float
    row: int = 0
    column: int = 0


@dataclass
class SubfigureDirective:
    """Rendering instructions for a LaTeX subfigure."""

    image_index: int
    trim: Tuple[int, int, int, int]
    width_ratio: float
    caption: str


@dataclass
class FigureVisionProfile:
    """Vision features computed from a figure asset."""

    owner_id: str
    image_path: str
    width: int
    height: int
    figure_type: str
    panel_boxes: List[PanelBox] = field(default_factory=list)
    colorfulness: float = 0.0
    edge_density: float = 0.0

    @property
    def panel_count(self) -> int:
        return len(self.panel_boxes)


@dataclass
class FigureRenderPlan:
    """Aggregated decision for how a block should render its figures."""

    caption: str
    float_spec: str
    width: float
    classification: str
    panel_count: int
    subfigures: List[SubfigureDirective] = field(default_factory=list)
    caption_score: float = 0.0
    quality_breakdown: Dict[str, float] = field(default_factory=dict)
    label_hint: str | None = None


@dataclass
class TableAssessment:
    """Lightweight summary for table complexity heuristics."""

    chunk_id: str
    columns: int
    rows: int
    complexity: float
    recommended_width: float


class FigureTableAgent:
    """Detect subfigures, classify figures, and plan placement/captions."""

    def __init__(
        self,
        llm_refiner: "LLMRefiner" | None = None,
        *,
        min_panel_area_ratio: float = 0.04,
    ) -> None:
        self.llm_refiner = llm_refiner
        self.min_panel_area_ratio = min_panel_area_ratio
        self._profiles_by_owner: Dict[str, List[FigureVisionProfile]] = {}
        self._table_assessments: Dict[str, TableAssessment] = {}
        self._package_requests: Dict[str, Optional[str]] = {}

    # ------------------------------ public API ------------------------------
    def analyze_plan(self, plan: Sequence[common.PlanBlock]) -> None:
        """Pre-compute CV features for every figure/table block."""

        for block in plan:
            if block.block_type == "figure" and block.images:
                self._profiles_by_owner[block.block_id] = []
                for image_path in block.images:
                    profile = self._analyze_image(block.block_id, image_path)
                    if profile:
                        self._profiles_by_owner[block.block_id].append(profile)
                        if profile.panel_count > 1:
                            self._request_package("subcaption")
            if block.block_type == "table":
                self._table_assessments[block.chunk_id] = self._assess_table_block(block)

    def prepare_render_plan(
        self,
        block: common.PlanBlock,
        snippet: str,
        chunk_text: str,
        references: ReferenceIndex | None,
    ) -> FigureRenderPlan:
        profiles = list(self._profiles_by_owner.get(block.block_id, []))
        if not profiles and block.images:
            profiles = [profile for profile in (self._analyze_image(block.block_id, path) for path in block.images) if profile]
            if profiles:
                self._profiles_by_owner[block.block_id] = profiles
        panel_count = sum(profile.panel_count for profile in profiles)
        classification = self._aggregate_classification(profiles) or "figure"
        caption = self._generate_caption(block, snippet, chunk_text, classification, panel_count)
        caption_score, breakdown = self._score_caption(caption, chunk_text or snippet)
        float_spec = self._choose_float_spec(block, references, classification, panel_count)
        width = self._recommend_width(profiles, classification, panel_count)
        subfigures = self._build_subfigure_directives(block, profiles) if panel_count > 1 else []
        if subfigures:
            self._request_package("subcaption")
        label_hint = references.label_for_block(block.block_id) if references else None
        plan = FigureRenderPlan(
            caption=caption,
            float_spec=float_spec,
            width=width,
            classification=classification,
            panel_count=panel_count,
            subfigures=subfigures,
            caption_score=caption_score,
            quality_breakdown=breakdown,
            label_hint=label_hint,
        )
        return plan

    def caption_from_chunk(self, chunk: common.Chunk) -> Tuple[str, float]:
        """Generate a caption/score pair for specialist usage."""

        metadata = chunk.metadata or {}
        figure_text = metadata.get("figure_caption") or chunk.text
        images = chunk.images or metadata.get("image_refs") or []
        panel_counts = 0
        profiles = []
        for path in images:
            profile = self._analyze_image(chunk.chunk_id, path)
            if profile:
                profiles.append(profile)
                panel_counts += profile.panel_count
        classification = self._aggregate_classification(profiles) or "figure"
        caption = self._generate_caption_from_context(figure_text, classification, panel_counts)
        score, _ = self._score_caption(caption, figure_text)
        return caption, score

    def assess_table(self, chunk: common.Chunk) -> TableAssessment:
        cached = self._table_assessments.get(chunk.chunk_id)
        if cached:
            return cached
        assessment = self._derive_table_assessment(chunk)
        self._table_assessments[chunk.chunk_id] = assessment
        return assessment

    def required_packages(self) -> List[Dict[str, str | None]]:
        return [{"package": name, "options": options} for name, options in self._package_requests.items()]

    # ------------------------------ internal helpers ------------------------------
    def _request_package(self, package: str, options: str | None = None) -> None:
        if package not in self._package_requests:
            self._package_requests[package] = options

    def _analyze_image(self, owner_id: str, image_path: str) -> FigureVisionProfile | None:
        try:
            resolved = str(Path(image_path))
        except Exception:
            return None
        existing = next(
            (profile for profile in self._profiles_by_owner.get(owner_id, []) if profile.image_path == resolved),
            None,
        )
        if existing:
            return existing
        if cv2 is None:
            profile = FigureVisionProfile(owner_id=owner_id, image_path=resolved, width=0, height=0, figure_type="figure")
            self._profiles_by_owner.setdefault(owner_id, []).append(profile)
            return profile
        image = cv2.imread(resolved)
        if image is None:
            LOGGER.debug("Figure agent could not read %s", resolved)
            return None
        height, width = image.shape[:2]
        boxes = self._detect_subfigures(image, width, height)
        figure_type = self._classify_figure(image, boxes)
        colorfulness = self._colorfulness(image)
        edge_density = self._edge_density(image)
        profile = FigureVisionProfile(
            owner_id=owner_id,
            image_path=resolved,
            width=width,
            height=height,
            figure_type=figure_type,
            panel_boxes=boxes,
            colorfulness=colorfulness,
            edge_density=edge_density,
        )
        self._profiles_by_owner.setdefault(owner_id, []).append(profile)
        return profile

    def _detect_subfigures(self, image, width: int, height: int) -> List[PanelBox]:
        if cv2 is None:
            return []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        inverted = 255 - thresh
        contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = width * height * self.min_panel_area_ratio
        boxes: List[PanelBox] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area < min_area:
                continue
            area_ratio = area / (width * height)
            boxes.append(PanelBox(x=x, y=y, w=w, h=h, area_ratio=area_ratio))
        if not boxes:
            return []
        row_threshold = max(12, int(height * 0.06))
        boxes.sort(key=lambda item: (item.y, item.x))
        rows: List[List[PanelBox]] = []
        for box in boxes:
            placed = False
            for row in rows:
                if abs(box.y - row[0].y) <= row_threshold:
                    row.append(box)
                    placed = True
                    break
            if not placed:
                rows.append([box])
        for row_idx, row in enumerate(rows):
            row.sort(key=lambda item: item.x)
            for col_idx, box in enumerate(row):
                box.row = row_idx
                box.column = col_idx
        return boxes

    def _classify_figure(self, image, boxes: List[PanelBox]) -> str:
        if cv2 is None:
            return "figure"
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
        white_ratio = float(np.sum(binary == 255)) / binary.size
        edge = self._edge_density(image)
        color = self._colorfulness(image)
        if len(boxes) >= 3:
            return "multi-panel"
        if edge > 0.22 and white_ratio < 0.75:
            return "plot"
        if color > 25 and edge < 0.18:
            return "photo"
        if white_ratio < 0.35:
            return "screenshot"
        return "diagram"

    def _aggregate_classification(self, profiles: Sequence[FigureVisionProfile]) -> str | None:
        if not profiles:
            return None
        ranking = ["multi-panel", "plot", "diagram", "photo", "screenshot", "figure"]
        scores = {label: 0 for label in ranking}
        for profile in profiles:
            scores[profile.figure_type] = scores.get(profile.figure_type, 0) + 1
        scores_sorted = sorted(scores.items(), key=lambda item: (-item[1], ranking.index(item[0])))
        return scores_sorted[0][0]

    def _generate_caption(
        self,
        block: common.PlanBlock,
        snippet: str,
        chunk_text: str,
        classification: str,
        panel_count: int,
    ) -> str:
        baseline = (
            block.metadata.get("figure_caption")
            or (snippet.splitlines()[0] if snippet.strip() else "")
            or chunk_text.splitlines()[0:1][0] if chunk_text.strip() else ""
            or block.label
            or "Auto-captioned figure"
        )
        enriched = self._generate_caption_from_context(baseline, classification, panel_count)
        return self._refine_caption(enriched, classification, chunk_text)

    def _generate_caption_from_context(self, context: str, classification: str, panel_count: int) -> str:
        sentence = context.strip().split("\n")[0] if context else ""
        fragments = [sentence or f"{classification.title()} overview"]
        if panel_count > 1:
            fragments.append(f"composed of {panel_count} panels")
        if classification not in fragments[0].lower():
            fragments.append(f"({classification})")
        return " ".join(fragments).strip().rstrip(".") + "."

    def _refine_caption(self, caption: str, classification: str, chunk_text: str) -> str:
        refiner = getattr(self, "llm_refiner", None)
        if not refiner:
            return caption
        context = f"Figure type: {classification}. Context: {chunk_text[:280]}"
        try:
            refined = refiner.refine("default", context, caption, None, None)
        except Exception as exc:  # pragma: no cover - LLM runtime dependent
            LOGGER.debug("Caption refinement skipped: %s", exc)
            return caption
        return self._extract_caption_text(refined) or caption

    def _extract_caption_text(self, payload: str) -> str:
        text = payload.strip()
        if not text:
            return ""
        if "\\caption" in text:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                return text[start + 1 : end].strip()
        return text

    def _score_caption(self, caption: str, context: str) -> Tuple[float, Dict[str, float]]:
        if not caption:
            return 0.0, {"descriptiveness": 0.0, "relevance": 0.0}
        tokens = caption.lower().split()
        unique_tokens = set(tokens)
        descriptiveness = min(1.0, len(unique_tokens) / 12)
        overlap = unique_tokens.intersection(set(context.lower().split())) if context else set()
        relevance = min(1.0, (len(overlap) / max(1, len(unique_tokens))) + 0.2)
        quality = {"descriptiveness": round(descriptiveness, 3), "relevance": round(relevance, 3)}
        return round((descriptiveness * 0.6 + relevance * 0.4), 3), quality

    def _choose_float_spec(
        self,
        block: common.PlanBlock,
        references: ReferenceIndex | None,
        classification: str,
        panel_count: int,
    ) -> str:
        label = references.label_for_block(block.block_id) if references else None
        ref_count = 0
        if label and references:
            ref_count = sum(1 for ref in references.references if ref.resolved_label == label)
        if panel_count > 1:
            return "p"
        if classification in {"plot", "diagram"} or ref_count >= 3:
            return "htbp"
        return "H"

    def _recommend_width(
        self,
        profiles: Sequence[FigureVisionProfile],
        classification: str,
        panel_count: int,
    ) -> float:
        if panel_count > 1:
            return 0.95
        if classification == "photo":
            return 0.78
        if not profiles:
            return 0.85
        avg_edge = sum(profile.edge_density for profile in profiles) / max(1, len(profiles))
        avg_color = sum(profile.colorfulness for profile in profiles) / max(1, len(profiles))
        complexity = avg_edge * 1.2 + avg_color * 0.01
        width = 0.82 + complexity * 0.1
        return max(0.7, min(0.95, width))

    def _build_subfigure_directives(
        self,
        block: common.PlanBlock,
        profiles: Sequence[FigureVisionProfile],
    ) -> List[SubfigureDirective]:
        if not profiles:
            return []
        index_lookup = {str(Path(path)): idx for idx, path in enumerate(block.images)}
        directives: List[SubfigureDirective] = []
        panel_idx = 0
        for profile in profiles:
            if not profile.panel_boxes or profile.width == 0 or profile.height == 0:
                continue
            row_counts: Dict[int, int] = {}
            for box in profile.panel_boxes:
                row_counts[box.row] = row_counts.get(box.row, 0) + 1
            for box in profile.panel_boxes:
                columns = max(1, row_counts.get(box.row, 1))
                width_ratio = min(0.95 / columns, 0.48)
                trim = self._panel_trim(profile, box)
                caption = f"Panel {chr(97 + (panel_idx % 26))}"
                directives.append(
                    SubfigureDirective(
                        image_index=index_lookup.get(profile.image_path, 0),
                        trim=trim,
                        width_ratio=round(width_ratio, 3),
                        caption=caption,
                    )
                )
                panel_idx += 1
        return directives

    def _panel_trim(self, profile: FigureVisionProfile, box: PanelBox) -> Tuple[int, int, int, int]:
        left = max(0, box.x)
        top = max(0, box.y)
        right = max(0, profile.width - (box.x + box.w))
        bottom = max(0, profile.height - (box.y + box.h))
        return (left, bottom, right, top)

    def _assess_table_block(self, block: common.PlanBlock) -> TableAssessment:
        metadata = block.metadata or {}
        if isinstance(metadata, dict):
            raw_signature = metadata.get("table_signature")
            signature = raw_signature if isinstance(raw_signature, dict) else {}
        else:
            signature = {}
        rows = signature.get("rows", 1)
        cols = signature.get("columns", 1)
        complexity = rows * cols
        width = max(0.65, min(0.95, 0.6 + cols * 0.05))
        return TableAssessment(
            chunk_id=block.chunk_id,
            columns=cols,
            rows=rows,
            complexity=float(complexity),
            recommended_width=width,
        )

    def _derive_table_assessment(self, chunk: common.Chunk) -> TableAssessment:
        meta = chunk.metadata or {}
        signature = meta.get("table_signature") or {}
        rows = signature.get("rows") or chunk.text.count("\\n") or 1
        columns = signature.get("columns") or max(1, len(chunk.text.split("|")) - 1)
        complexity = rows * columns
        width = max(0.65, min(0.9, 0.55 + columns * 0.04))
        return TableAssessment(
            chunk_id=chunk.chunk_id,
            columns=columns,
            rows=rows,
            complexity=float(complexity),
            recommended_width=width,
        )

    def _colorfulness(self, image) -> float:
        if np is None:
            return 0.0
        (b, g, r) = cv2.split(image)
        rg = np.absolute(r - g)
        yb = np.absolute(0.5 * (r + g) - b)
        return float(np.mean(np.sqrt(rg ** 2 + yb ** 2)))

    def _edge_density(self, image) -> float:
        if cv2 is None:
            return 0.0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return float(np.count_nonzero(edges)) / edges.size

    def to_json(self, block_id: str) -> str:
        profiles = self._profiles_by_owner.get(block_id) or []
        return json.dumps(
            [
                {
                    "image_path": profile.image_path,
                    "figure_type": profile.figure_type,
                    "panel_count": profile.panel_count,
                }
                for profile in profiles
            ],
            indent=2,
        )


__all__ = [
    "FigureTableAgent",
    "FigureRenderPlan",
    "PanelBox",
    "SubfigureDirective",
    "TableAssessment",
]
