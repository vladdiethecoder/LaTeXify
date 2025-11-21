"""Evaluator agent that scores snippets and decides PASS/FAIL."""
from __future__ import annotations

from dataclasses import dataclass

from .graph_state import GraphState


@dataclass
class EvaluatorAgent:
    """Tiny heuristic evaluator; replace with LM scoring later."""

    min_length: int = 20

    def evaluate(self, state: GraphState) -> GraphState:
        latex = state.candidate_latex or ""
        if len(latex) < self.min_length:
            state.evaluation = "FAIL"
            state.score_notes = "snippet too short"
        elif "auto_repaired" == state.diagnostics:
            state.evaluation = "FAIL"
            state.score_notes = "requires manual inspection"
        else:
            state.evaluation = "PASS"
            state.score_notes = "meets heuristic constraints"
        state.mark_stage("evaluator", notes=state.evaluation or "")
        state.record_metrics(evaluator_length=len(latex))
        state.log(f"evaluator: {state.evaluation} ({state.score_notes})")
        return state


__all__ = ["EvaluatorAgent"]
