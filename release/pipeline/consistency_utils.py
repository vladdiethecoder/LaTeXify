"""Combined consistency utilities (symbolic + multimodal)."""
from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple

try:  # pragma: no cover - optional heavy dependency
    import sympy as sp  # type: ignore
except Exception:  # pragma: no cover
    sp = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from latex2sympy2 import latex2sympy  # type: ignore
except Exception:  # pragma: no cover
    latex2sympy = None  # type: ignore

from PIL import Image

from ..core import common
from .ingestion import ClipCaptionVerifier
from .reward_suite import aesthetic_mm_score

LOGGER = logging.getLogger(__name__)
VARIABLE_RE = re.compile(r"[A-Za-z]\\w*")
OPERATOR_SET = {"+", "-", "*", "/", "^", "=", r"\\times", r"\\cdot"}


@dataclass
class MathConsistencyValidator:
    """Compare symbolic content between source OCR text and generated LaTeX."""

    def variable_overlap(self, source: str, generated: str) -> float:
        source_vars = self._extract_variables(source)
        generated_vars = self._extract_variables(generated)
        if not source_vars or not generated_vars:
            return 0.0
        intersection = len(source_vars & generated_vars)
        union = len(source_vars | generated_vars)
        return intersection / union if union else 0.0

    def operator_overlap(self, source: str, generated: str) -> float:
        source_ops = self._extract_operators(source)
        generated_ops = self._extract_operators(generated)
        if not source_ops or not generated_ops:
            return 0.0
        intersection = len(source_ops & generated_ops)
        union = len(source_ops | generated_ops)
        return intersection / union if union else 0.0

    def structure_similarity(self, source: str, generated: str) -> float:
        lhs = self._to_sympy(source)
        rhs = self._to_sympy(generated)
        if lhs is None or rhs is None:
            return 0.0
        try:  # pragma: no cover - sympy heavy
            simplified = sp.simplify(lhs - rhs) if sp is not None else None
            if simplified == 0:
                return 1.0
        except Exception:
            return 0.0
        return 0.0

    def validate(self, source: str, generated: str) -> Dict[str, float]:
        return {
            "symbol_overlap": round(self.variable_overlap(source, generated), 3),
            "operator_overlap": round(self.operator_overlap(source, generated), 3),
            "structure_similarity": round(self.structure_similarity(source, generated), 3),
        }

    def _extract_variables(self, text: str) -> Set[str]:
        return {match.group(0) for match in VARIABLE_RE.finditer(text or "")}

    def _extract_operators(self, text: str) -> Set[str]:
        operators = set()
        for op in OPERATOR_SET:
            if op in text:
                operators.add(op)
        return operators

    def _to_sympy(self, text: str):
        if not text.strip():
            return None
        if latex2sympy is not None:
            try:
                return latex2sympy(text)
            except Exception:
                pass
        if sp is not None:
            try:
                return sp.sympify(text)
            except Exception:
                return None
        return None


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
    """Runs multimodal checks linking rendered visuals to semantics."""

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
        if pdf_path.exists():
            score = aesthetic_mm_score(tex_path)
        else:
            score = 0.0
        suggestions: List[str] = []
        if score < 0.5:
            suggestions.append("Overall layout quality is low; ensure consistent margins and spacing.")
        elif score < 0.75:
            suggestions.append("Layout acceptable but could benefit from tighter alignment and spacing adjustments.")
        return LayoutConsistency(score=round(score, 3), suggestions=suggestions)


def visual_textual_consistency(
    tex_path: Path,
    plan_path: Path,
    chunks_path: Path,
    output_path: Path | None = None,
    assets_dir: Path | None = None,
) -> Dict[str, object]:
    """Helper wrapper used by reward utilities to dump a consistency report."""

    plan = common.load_plan(plan_path) if plan_path.exists() else []
    chunk_map = {chunk.chunk_id: chunk for chunk in common.load_chunks(chunks_path)} if chunks_path.exists() else {}
    verifier = ConsistencyVerifier()
    assets = assets_dir or tex_path.parent / "assets"
    report = verifier.evaluate(tex_path, plan, chunk_map, assets)
    payload = report.to_json()
    target = output_path or tex_path.parent / "consistency.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


__all__ = [
    "MathConsistencyValidator",
    "ConsistencyVerifier",
    "ConsistencyReport",
    "FigureConsistency",
    "EquationConsistency",
    "LayoutConsistency",
    "visual_textual_consistency",
]
