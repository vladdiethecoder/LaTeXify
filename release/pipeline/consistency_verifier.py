"""Visual-textual consistency verification across figures, equations, and layout."""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image

from ..core import common
from .reward_mm import aesthetic_mm_score
from .ingestion import ClipCaptionVerifier

LOGGER = logging.getLogger(__name__)


@dataclass
class FigureConsistency:
    chunk_id: str
    caption: str
    score: float
    image_path: str | None
    note: str | None = None


@dataclass
class EquationConsistency:
    chunk_id: str
    balanced: bool
    issues: List[str] = field(default_factory=list)


@dataclass
class LayoutConsistency:
    score: float
    suggestions: List[str] = field(default_factory=list)


@dataclass
class ConsistencyReport:
    figure_results: List[FigureConsistency]
    equation_results: List[EquationConsistency]
    layout: LayoutConsistency

    def to_json(self) -> Dict[str, object]:
        return {
            "figures": [vars(entry) for entry in self.figure_results],
            "equations": [vars(entry) for entry in self.equation_results],
            "layout": {
                "score": self.layout.score,
                "suggestions": self.layout.suggestions,
            },
        }


class ConsistencyVerifier:
    """Runs multimodal checks that link rendered visuals to semantics."""

    def __init__(self) -> None:
        self.caption_verifier = ClipCaptionVerifier()

    def evaluate(
        self,
        tex_path: Path,
        plan: List[common.PlanBlock],
        chunk_map: Dict[str, common.Chunk],
        assets_dir: Path,
    ) -> ConsistencyReport:
        figure_results = self._check_figures(plan, chunk_map, assets_dir)
        equation_results = self._check_equations(plan, chunk_map)
        layout = self._check_layout(tex_path)
        return ConsistencyReport(
            figure_results=figure_results,
            equation_results=equation_results,
            layout=layout,
        )

    def _check_figures(
        self,
        plan: List[common.PlanBlock],
        chunk_map: Dict[str, common.Chunk],
        assets_dir: Path,
    ) -> List[FigureConsistency]:
        results: List[FigureConsistency] = []
        for block in plan:
            if block.block_type != "figure":
                continue
            chunk = chunk_map.get(block.chunk_id)
            caption = chunk.metadata.get("figure_caption") if chunk else block.label
            caption = caption or block.label or ""
            image_path = None
            for candidate in block.images:
                candidate_path = Path(candidate)
                if candidate_path.exists():
                    image_path = str(candidate_path)
                    break
            if not image_path:
                # try assets directory copy
                for asset in assets_dir.glob("*"):
                    if asset.name in {Path(img).name for img in block.images}:
                        image_path = str(asset)
                        break
            if not caption.strip():
                results.append(FigureConsistency(block.chunk_id, caption, 0.0, image_path, "missing caption"))
                continue
            score = self.caption_verifier.score(image_path, caption)
            note = None
            if score < 0.15:
                note = "Caption likely mismatched with figure"
            results.append(FigureConsistency(block.chunk_id, caption, score, image_path, note))
        return results

    def _check_equations(
        self,
        plan: List[common.PlanBlock],
        chunk_map: Dict[str, common.Chunk],
    ) -> List[EquationConsistency]:
        results: List[EquationConsistency] = []
        for block in plan:
            if block.block_type != "equation":
                continue
            chunk = chunk_map.get(block.chunk_id)
            text = chunk.text if chunk else ""
            balanced, issues = self._equation_health(text)
            results.append(EquationConsistency(chunk_id=block.chunk_id, balanced=balanced, issues=issues))
        return results

    def _equation_health(self, text: str) -> Tuple[bool, List[str]]:
        stack = []
        issues: List[str] = []
        for char in text:
            if char == "{":
                stack.append(char)
            elif char == "}":
                if not stack:
                    issues.append("extra closing brace")
                    break
                stack.pop()
        if stack:
            issues.append("unmatched braces")
        if text.count("\\left") != text.count("\\right"):
            issues.append("left/right delimiter mismatch")
        if "?" in text:
            issues.append("placeholder '?' detected in equation")
        return (not issues), issues

    def _check_layout(self, tex_path: Path) -> LayoutConsistency:
        pdf_path = tex_path.with_suffix(".pdf")
        if not pdf_path.exists():
            compile_tex = None
            try:  # local import to avoid circular dependency
                from .assembly import compile_tex as _compile

                compile_tex = _compile
            except Exception:
                compile_tex = None
            if compile_tex is not None:
                compile_tex(tex_path)
        score = aesthetic_mm_score(tex_path) if pdf_path.exists() else 0.0
        suggestions: List[str] = []
        if score < 0.5:
            suggestions.append("Overall layout quality is low; ensure consistent margins and spacing.")
        elif score < 0.75:
            suggestions.append("Layout acceptable but could benefit from tighter alignment and spacing adjustments.")
        return LayoutConsistency(score=round(score, 3), suggestions=suggestions)


__all__ = ["ConsistencyVerifier", "ConsistencyReport"]
