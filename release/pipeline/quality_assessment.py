"""Composite quality scoring for generated LaTeX documents."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Dict, List

from ..core import common
from .consistency import MathConsistencyValidator
from .math_environment import MathEnvironmentDetector


@dataclass
class QualityAssessor:
    """Compute math, structural, and stylistic quality metrics."""

    target_score: float = 0.75
    validator: MathConsistencyValidator = field(default_factory=MathConsistencyValidator)
    env_detector: MathEnvironmentDetector = field(default_factory=MathEnvironmentDetector)

    def evaluate(
        self,
        tex_path: Path,
        chunks_path: Path | None,
        plan_path: Path,
        snippets_path: Path,
    ) -> Dict[str, object]:
        tex = tex_path.read_text(encoding="utf-8") if tex_path.exists() else ""
        chunks = {chunk.chunk_id: chunk for chunk in common.load_chunks(chunks_path)} if chunks_path and chunks_path.exists() else {}
        snippets = {snippet.chunk_id: snippet for snippet in common.load_snippets(snippets_path)} if snippets_path.exists() else {}
        plan = common.load_plan(plan_path)
        math_scores: List[float] = []
        weak_sections: List[str] = []
        aesthetics_hits = 0
        aesthetic_candidates = 0
        semantic_scores: List[float] = []
        for block in plan:
            chunk = chunks.get(block.chunk_id)
            snippet = snippets.get(block.chunk_id)
            if not chunk or not snippet:
                continue
            stats = self.validator.validate(chunk.text, snippet.latex)
            math_scores.append(stats["symbol_overlap"])
            if stats["symbol_overlap"] < 0.4:
                weak_sections.append(block.chunk_id)
            semantic_scores.append(self._semantic_similarity(chunk.text, snippet.latex))
            if block.block_type in {"equation", "question", "problem"}:
                aesthetic_candidates += 1
                env = self.env_detector.detect(block.block_type, snippet.latex, chunk.metadata)
                if env in {"align", "align*", "cases", "bmatrix"}:
                    aesthetics_hits += 1
        math_preservation = mean(math_scores) if math_scores else 0.0
        structure_score = self._structure_score(tex, plan)
        syntax_score = self._syntax_score(tex)
        semantic_accuracy = mean(semantic_scores) if semantic_scores else 0.0
        aesthetics = aesthetics_hits / max(1, aesthetic_candidates)
        report = {
            "math_preservation": round(math_preservation, 3),
            "structure_fidelity": round(structure_score, 3),
            "syntactic_correctness": round(syntax_score, 3),
            "semantic_accuracy": round(semantic_accuracy, 3),
            "aesthetics": round(aesthetics, 3),
        }
        aggregate = mean(report.values()) if report else 0.0
        report["aggregate"] = round(aggregate, 3)
        report["weak_sections"] = sorted(set(weak_sections))
        return report

    def _structure_score(self, tex: str, plan: List[common.PlanBlock]) -> float:
        sections = [block for block in plan if block.block_type == "section"]
        if not sections:
            return 1.0
        hits = 0
        for block in sections:
            if not block.label:
                continue
            if block.label in tex:
                hits += 1
        return hits / max(1, len(sections))

    def _syntax_score(self, tex: str) -> float:
        stack = 0
        for char in tex:
            if char == "{":
                stack += 1
            elif char == "}":
                stack = max(0, stack - 1)
        return 1.0 if stack == 0 else max(0.0, 1.0 - min(1.0, stack / 10))

    def _semantic_similarity(self, source: str, generated: str) -> float:
        if not source.strip() or not generated.strip():
            return 0.0
        ratio = len(generated.strip()) / max(1, len(source.strip()))
        if ratio > 1.5 or ratio < 0.5:
            return 0.2
        return min(1.0, 1.0 - abs(1 - ratio))


__all__ = ["QualityAssessor"]
