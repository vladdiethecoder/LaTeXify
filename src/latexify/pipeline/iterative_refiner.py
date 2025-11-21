"""Iterative refinement loop driven by quality metrics."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List

from ..core import common
from .math_support import MathEnvironmentDetector
from .quality_assessment import QualityAssessor
from .symbol_normalizer import SymbolNormalizer
from .assembly import ProgressiveAssembler
from .latex_repair_agent import KimiK2LatexRepair
from .robust_compilation import run_robust_compilation


@dataclass
class IterativeRefinerResult:
    tex_path: Path
    report: Dict[str, object]
    iterations: int


@dataclass
class IterativeRefiner:
    """Re-run weak sections until the document meets quality thresholds."""

    assessor: QualityAssessor
    plan_path: Path
    snippets_path: Path
    chunks_path: Path | None
    preamble_path: Path | None
    output_dir: Path
    title: str
    author: str
    sanitize_unicode: bool = True
    skip_compile: bool = False
    max_iterations: int = 2
    symbol_normalizer: SymbolNormalizer = field(default_factory=SymbolNormalizer)
    env_detector: MathEnvironmentDetector = field(default_factory=MathEnvironmentDetector)
    repair_agent: KimiK2LatexRepair = field(default_factory=KimiK2LatexRepair)
    _plan_cache: Dict[str, common.PlanBlock] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self._plan_cache = {}

    def refine(self) -> IterativeRefinerResult:
        tex_path = None
        report: Dict[str, object] = {}
        for iteration in range(self.max_iterations):
            assembler = ProgressiveAssembler(
                plan_path=self.plan_path,
                snippets_path=self.snippets_path,
                preamble_path=self.preamble_path,
                output_dir=self.output_dir,
                title=self.title,
                author=self.author,
                chunks_path=self.chunks_path,
                symbol_normalizer=self.symbol_normalizer,
            )
            tex_path = assembler.build(sanitize_unicode=self.sanitize_unicode)
            if not self.skip_compile:
                self.repair_agent.preflight_check(tex_path)
                result = run_robust_compilation(
                    tex_path,
                    cache_dir=tex_path.parent / ".latexify_cache",
                    repair_agent=self.repair_agent,
                    max_retries=self.max_iterations,
                )
                if result.pdf_path is None:
                    log_path = tex_path.with_suffix(".log")
                    if assembler.error_repair.repair(tex_path, log_path):
                        run_robust_compilation(
                            tex_path,
                            cache_dir=tex_path.parent / ".latexify_cache",
                            repair_agent=self.repair_agent,
                            max_retries=self.max_iterations,
                        )
            report = self.assessor.evaluate(tex_path, self.chunks_path, self.plan_path, self.snippets_path)
            if report.get("aggregate", 0.0) >= self.assessor.target_score or not report.get("weak_sections"):
                return IterativeRefinerResult(tex_path=tex_path, report=report, iterations=iteration + 1)
            self._strengthen_snippets(report["weak_sections"])
        if tex_path is None:
            raise RuntimeError("IterativeRefiner was unable to build an initial document.")
        return IterativeRefinerResult(tex_path=tex_path, report=report, iterations=self.max_iterations)

    def _strengthen_snippets(self, chunk_ids: Iterable[str]) -> None:
        snippets = {snippet.chunk_id: snippet for snippet in common.load_snippets(self.snippets_path)}
        plan_map = self._load_plan_map()
        updated = False
        for chunk_id in chunk_ids:
            snippet = snippets.get(chunk_id)
            block = plan_map.get(chunk_id)
            if not snippet or not block:
                continue
            normalized = self.symbol_normalizer.normalize(snippet.latex)
            metadata = block.metadata or {}
            wrapped = self.env_detector.wrap(block.block_type or "text", normalized, metadata)
            if wrapped != snippet.latex:
                snippet.latex = wrapped
                updated = True
        if updated:
            common.save_snippets(snippets.values(), self.snippets_path)

    def _load_plan_map(self) -> Dict[str, common.PlanBlock]:
        if self._plan_cache:
            return self._plan_cache
        if self.plan_path.exists():
            self._plan_cache = {block.chunk_id: block for block in common.load_plan(self.plan_path)}
        else:
            self._plan_cache = {}
        return self._plan_cache


__all__ = ["IterativeRefiner", "IterativeRefinerResult"]
