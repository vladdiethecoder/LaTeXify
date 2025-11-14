"""Compile-and-repair agent that emulates the compile loop."""
from __future__ import annotations

from dataclasses import dataclass

from .graph_state import GraphState


@dataclass
class CompileAndRepairAgent:
    """Simulate a compile attempt and apply simple structural repairs."""

    auto_wrap_equations: bool = True

    def run(self, state: GraphState) -> GraphState:
        latex = state.candidate_latex or ""
        balanced = latex.count("\\begin") == latex.count("\\end")
        if self.auto_wrap_equations and "\\begin" not in latex:
            latex = "\\begin{equation}\n" + latex + "\n\\end{equation}"
        if not balanced:
            latex += "\n\\end{aligned}"
        state.candidate_latex = latex
        state.diagnostics = "compile_ok" if balanced else "auto_repaired"
        state.log(f"compile: diagnostics={state.diagnostics}")
        return state


__all__ = ["CompileAndRepairAgent"]
