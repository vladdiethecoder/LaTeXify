"""Compile-and-repair agent that emulates the compile loop."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from textwrap import dedent
from typing import Optional

from ..models.local_llm import LocalLLMConfig, LocalLLMOrchestrator, LocalLLMUnavailable
from .graph_state import GraphState

REPAIR_PROMPT = dedent(
    """
    You repair LaTeX snippets. When given the chunk text and a broken LaTeX candidate,
    return only the corrected LaTeX wrapped in <latex>...</latex>. Preserve math semantics,
    fix missing \\begin/\\end pairs, and keep the style concise.

    <chunk>
    {chunk_text}
    </chunk>
    <candidate>
    {candidate}
    </candidate>
    <latex>
    """
).strip()


def _discover_default_llm() -> Optional[Path]:
    search_root = Path(__file__).resolve().parents[1] / "models" / "llm"
    if not search_root.exists():
        return None
    for candidate in sorted(search_root.rglob("*.gguf")):
        if candidate.is_file():
            return candidate
    return None


@dataclass
class CompileAndRepairAgent:
    """Compile the snippet and apply structural repairs via a local LLM when needed."""

    auto_wrap_equations: bool = True
    repair_max_tokens: int = 384
    repair_temperature: float = 0.15
    repair_model_path: str | None = None
    repair_grammar_path: str | None = None
    _llm: LocalLLMOrchestrator | None = field(default=None, init=False, repr=False)
    _llm_failed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.repair_model_path is None:
            override = os.environ.get("LATEXIFY_REPAIR_LLM_MODEL")
            if override:
                self.repair_model_path = override
        if self.repair_grammar_path is None:
            self.repair_grammar_path = os.environ.get("LATEXIFY_REPAIR_LLM_GRAMMAR")
        if self.repair_model_path is None:
            default_model = _discover_default_llm()
            if default_model:
                self.repair_model_path = str(default_model)

    @staticmethod
    def _has_explicit_math_wrapper(payload: str) -> bool:
        stripped = payload.strip()
        if not stripped:
            return False
        return stripped.startswith(("\\begin", "\\[", "\\(", "$$"))

    def _ensure_llm(self) -> LocalLLMOrchestrator | None:
        if self._llm_failed:
            return None
        if self._llm is not None:
            return self._llm
        if not self.repair_model_path:
            self._llm_failed = True
            return None
        model_path = Path(self.repair_model_path).expanduser()
        if not model_path.exists():
            self._llm_failed = True
            return None
        config = LocalLLMConfig(
            model_path=model_path,
            grammar_path=Path(self.repair_grammar_path).expanduser()
            if self.repair_grammar_path
            else None,
        )
        try:
            self._llm = LocalLLMOrchestrator(config)
        except (LocalLLMUnavailable, OSError, RuntimeError):
            self._llm_failed = True
            self._llm = None
        return self._llm

    def _invoke_llm_repair(self, chunk_text: str, snippet: str) -> str | None:
        engine = self._ensure_llm()
        if engine is None:
            return None
        prompt = REPAIR_PROMPT.format(
            chunk_text=chunk_text.strip() or "[no chunk context]",
            candidate=snippet.strip() or "[empty candidate]",
        )
        raw = engine.generate(prompt, max_tokens=self.repair_max_tokens, temperature=self.repair_temperature)
        cleaned = raw.strip()
        if not cleaned:
            return None
        if cleaned.startswith("<latex>"):
            cleaned = cleaned[len("<latex>") :]
        return cleaned.strip()

    def _heuristic_repair(self, latex: str, balanced: bool) -> tuple[str, bool, str]:
        diagnostics = "compile_ok"
        already_wrapped = self._has_explicit_math_wrapper(latex)
        if self.auto_wrap_equations and not already_wrapped:
            latex = "\\begin{equation}\n" + latex + "\n\\end{equation}"
            diagnostics = "auto_wrapped"
        if not balanced:
            latex += "\n\\end{aligned}"
            diagnostics = "auto_repaired"
            balanced = True
        return latex, balanced, diagnostics

    def run(self, state: GraphState) -> GraphState:
        latex = state.candidate_latex or ""
        chunk_context = state.content or ""
        balanced = latex.count("\\begin") == latex.count("\\end")
        needs_llm = not latex.strip() or not balanced
        if needs_llm:
            repaired = self._invoke_llm_repair(chunk_context, latex)
            if repaired:
                latex = repaired
                balanced = latex.count("\\begin") == latex.count("\\end")
                state.diagnostics = "llm_repaired"
        if not state.diagnostics:
            latex, balanced, diag = self._heuristic_repair(latex, balanced)
            state.diagnostics = diag
        state.candidate_latex = latex
        state.log(f"compile: diagnostics={state.diagnostics}")
        return state


__all__ = ["CompileAndRepairAgent"]
