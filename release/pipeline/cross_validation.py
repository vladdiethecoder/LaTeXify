"""Multi-modal cross validation that augments reward scoring."""
from __future__ import annotations

import math
import os
import re
from pathlib import Path
from dataclasses import dataclass
from time import perf_counter
from typing import Dict, Iterable, List, Sequence

from ..core import common
from ..models.kimi_k2_adapter import get_kimi_adapter
from . import kimi_metrics


LATEX_SECTION_RE = re.compile(r"\\section\{|\\subsection\{", re.IGNORECASE)
LATEX_MATH_RE = re.compile(r"\\begin\{(equation|align|gather)", re.IGNORECASE)
LATEX_TABLE_RE = re.compile(r"\\begin\{table", re.IGNORECASE)
LATEX_FIGURE_RE = re.compile(r"\\includegraphics", re.IGNORECASE)
TOK_SPLIT_RE = re.compile(r"[^A-Za-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return [token for token in TOK_SPLIT_RE.split(text or "") if token]


def _kimi_semantic_temperature() -> float:
    try:
        return float(os.environ.get("LATEXIFY_KIMI_K2_TEMPERATURE", "0.05"))
    except ValueError:
        return 0.05


@dataclass
class BranchConsistencyValidator:
    """Measures how many chunks preserved their vision branch provenance."""

    def score(self, chunks: Sequence[common.Chunk]) -> Dict[str, float]:
        if not chunks:
            return {"coverage": 0.0, "consistency": 1.0}
        linked = 0
        mismatched = 0
        confidences: List[float] = []
        for chunk in chunks:
            metadata = chunk.metadata or {}
            provenance = metadata.get("branch_provenance") or {}
            vision = provenance.get("vision")
            region = str(metadata.get("region_type", "")).lower()
            layout_conf = float(metadata.get("layout_confidence", 0.7) or 0.7)
            if isinstance(vision, dict):
                linked += 1
                branch_region = str(vision.get("region_type", "")).lower()
                if branch_region and branch_region != region:
                    mismatched += 1
                confidences.append(layout_conf)
            elif region in {"figure", "table", "equation"}:
                mismatched += 1
        coverage = linked / len(chunks)
        consistency = 1.0 if linked == 0 else max(0.0, 1.0 - mismatched / max(linked, 1))
        confidence = sum(confidences) / len(confidences) if confidences else 0.7
        return {
            "coverage": round(coverage, 3),
            "consistency": round(consistency, 3),
            "confidence": round(confidence, 3),
        }


@dataclass
class ContentPreservationScorer:
    """Compares source OCR text and generated LaTeX content."""

    def score(self, source_text: str, generated_text: str) -> Dict[str, float]:
        source_tokens = _tokenize(source_text)
        generated_tokens = _tokenize(generated_text)
        if not source_tokens or not generated_tokens:
            return {"token_overlap": 0.0, "length_ratio": 0.0}
        source_set = set(token.lower() for token in source_tokens)
        generated_set = set(token.lower() for token in generated_tokens)
        overlap = len(source_set & generated_set)
        union = len(source_set | generated_set)
        token_overlap = round(overlap / union, 3) if union else 0.0
        length_ratio = round(min(len(generated_tokens) / len(source_tokens), 1.0), 3)
        return {"token_overlap": token_overlap, "length_ratio": length_ratio}


@dataclass
class LayoutFidelityMetrics:
    """Structural metrics comparing chunk layout hints to final LaTeX."""

    def score(self, chunks: Sequence[common.Chunk], latex: str) -> Dict[str, float]:
        headings = sum(1 for chunk in chunks if (chunk.metadata or {}).get("header_level"))
        tables = sum(1 for chunk in chunks if (chunk.metadata or {}).get("region_type") == "table")
        equations = sum(1 for chunk in chunks if (chunk.metadata or {}).get("region_type") in {"equation", "formula"})
        layout_conf = [float((chunk.metadata or {}).get("layout_confidence", 0.7) or 0.7) for chunk in chunks]

        latex_headings = len(LATEX_SECTION_RE.findall(latex))
        latex_tables = len(LATEX_TABLE_RE.findall(latex))
        latex_math = len(LATEX_MATH_RE.findall(latex))

        return {
            "heading_alignment": _ratio(latex_headings, headings),
            "table_alignment": _ratio(latex_tables, tables),
            "math_alignment": _ratio(latex_math, equations),
            "layout_confidence": round(sum(layout_conf) / len(layout_conf), 3) if layout_conf else 0.7,
        }


@dataclass
class VisualTextConsistency:
    def score(self, chunks: Sequence[common.Chunk], latex: str, pdf_path: Path | None) -> Dict[str, float]:
        figure_chunks = sum(1 for chunk in chunks if (chunk.metadata or {}).get("region_type") == "figure")
        chunk_images = sum(len(chunk.images or []) for chunk in chunks)
        latex_figures = len(LATEX_FIGURE_RE.findall(latex))
        alignment = _ratio(latex_figures, figure_chunks)
        asset_coverage = _ratio(latex_figures, chunk_images) if chunk_images else alignment
        pdf_ready = 1.0 if pdf_path and pdf_path.exists() else 0.0
        return {
            "figure_alignment": alignment,
            "asset_coverage": asset_coverage,
            "pdf_available": pdf_ready,
        }


@dataclass
class SemanticKimiValidator:
    max_samples: int = 3

    def score(self, chunks: Sequence[common.Chunk], latex: str) -> Dict[str, float]:
        adapter = self._adapter()
        if not adapter:
            return {"kimi_consistency": 0.5, "confidence": 0.0, "samples": 0}
        scored = 0
        total = 0
        sampled = sorted(chunks, key=lambda chunk: float((chunk.metadata or {}).get("layout_confidence", 0.7)), reverse=True)
        for chunk in sampled[: self.max_samples]:
            prompt = self._build_prompt(chunk.text, latex)
            try:
                start = perf_counter()
                response = adapter.generate(
                    prompt,
                    max_tokens=80,
                    temperature=_kimi_semantic_temperature(),
                )
                kimi_metrics.record_inference(perf_counter() - start, bool(response))
            except Exception:
                kimi_metrics.record_inference(0.0, False)
                continue
            total += 1
            if response.lower().startswith("yes") or "supported" in response.lower():
                scored += 1
        if total == 0:
            return {"kimi_consistency": 0.5, "confidence": 0.0, "samples": 0}
        return {
            "kimi_consistency": round(scored / total, 3),
            "confidence": round(total / self.max_samples, 3),
            "samples": total,
        }

    def _build_prompt(self, source: str, latex: str) -> str:
        sample = latex[:2000]
        return (
            "Compare the source text with the LaTeX document."
            "Respond YES if the LaTeX contains the source ideas, otherwise NO.\n"
            f"Source:\n{source.strip()}\n\nLaTeX snippet:\n{sample}\n"
        )

    def _adapter(self):
        try:
            return get_kimi_adapter()
        except Exception:
            return None


def _ratio(output: int, expected: int) -> float:
    if expected == 0:
        return 1.0
    return round(min(output, expected) / expected, 3)


def _weighted_average(values: Iterable[float]) -> float:
    values = [value for value in values if value is not None]
    if not values:
        return 0.0
    return round(sum(values) / len(values), 3)


@dataclass
class CrossValidationReport:
    structural: Dict[str, float]
    content: Dict[str, float]
    visual: Dict[str, float]
    semantic: Dict[str, float]
    confidence: float
    overall_score: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "structural": self.structural,
            "content": self.content,
            "visual": self.visual,
            "semantic": self.semantic,
            "confidence": self.confidence,
            "overall_score": self.overall_score,
        }


def run_cross_validation(
    chunks: Sequence[common.Chunk],
    latex: str,
    validation_data: Dict[str, object] | None = None,
    *,
    pdf_path: Path | None = None,
    source_text: str | None = None,
) -> CrossValidationReport:
    branch_validator = BranchConsistencyValidator()
    branch_scores = branch_validator.score(chunks)
    preserv_scorer = ContentPreservationScorer()
    content_metrics = preserv_scorer.score(source_text or "\n\n".join(chunk.text for chunk in chunks), latex)
    layout_metrics = LayoutFidelityMetrics().score(chunks, latex)
    visual_scores = VisualTextConsistency().score(chunks, latex, pdf_path)
    semantic_scores = SemanticKimiValidator().score(chunks, latex)

    structural_score = _weighted_average([
        branch_scores.get("consistency", 1.0),
        branch_scores.get("coverage", 0.0),
        layout_metrics.get("heading_alignment", 1.0),
        layout_metrics.get("table_alignment", 1.0),
        layout_metrics.get("math_alignment", 1.0),
    ])
    content_score = _weighted_average([
        content_metrics.get("token_overlap", 0.0),
        content_metrics.get("length_ratio", 0.0),
    ])
    visual_score = _weighted_average([
        visual_scores.get("figure_alignment", 1.0),
        visual_scores.get("asset_coverage", 1.0),
        visual_scores.get("pdf_available", 0.0),
    ])
    semantic_score = semantic_scores.get("kimi_consistency", 0.5)
    confidence = _weighted_average([
        layout_metrics.get("layout_confidence", 0.7),
        semantic_scores.get("confidence", 0.0),
        branch_scores.get("confidence", 0.7),
    ])
    extra_penalty = 0.0
    if validation_data and validation_data.get("errors"):
        extra_penalty = 0.05
    overall = (0.3 * structural_score + 0.3 * content_score + 0.2 * visual_score + 0.2 * semantic_score)
    overall = max(0.0, min(1.0, overall * (0.85 + 0.15 * confidence) - extra_penalty))
    structural_payload = dict(branch_scores)
    structural_payload.update(layout_metrics)
    structural_payload["composite"] = round(structural_score, 3)
    content_payload = dict(content_metrics)
    content_payload["composite"] = round(content_score, 3)
    visual_payload = dict(visual_scores)
    visual_payload["composite"] = round(visual_score, 3)
    semantic_payload = dict(semantic_scores)
    semantic_payload["composite"] = round(semantic_score, 3)
    return CrossValidationReport(
        structural={k: round(v, 3) if isinstance(v, float) else v for k, v in structural_payload.items()},
        content=content_payload,
        visual={k: round(v, 3) if isinstance(v, float) else v for k, v in visual_payload.items()},
        semantic=semantic_payload,
        confidence=round(confidence, 3),
        overall_score=round(overall, 3),
    )


__all__ = [
    "BranchConsistencyValidator",
    "ContentPreservationScorer",
    "LayoutFidelityMetrics",
    "VisualTextConsistency",
    "SemanticKimiValidator",
    "CrossValidationReport",
    "run_cross_validation",
]
