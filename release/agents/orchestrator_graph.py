"""LangGraph-ready orchestration scaffold for the tiered agent stack."""
from __future__ import annotations

import argparse
import logging
from typing import Iterable, List, Sequence

from .compile_and_repair_agent import CompileAndRepairAgent
from .creative_agent import CreativeAgent
from .evaluator_agent import EvaluatorAgent
from .graph_state import ContentChunk, GraphState
from .research_agent import ResearchAgent

LOGGER = logging.getLogger(__name__)


def run_layout_graph(
    chunks: Sequence[ContentChunk],
    *,
    max_attempts: int = 4,
) -> List[GraphState]:
    """Iterate chunks through Creative→Compile→Evaluate, escalating to ResearchAgent."""

    creative = CreativeAgent()
    compiler = CompileAndRepairAgent()
    evaluator = EvaluatorAgent()
    researcher = ResearchAgent()
    completed: List[GraphState] = []

    for chunk in chunks:
        state = GraphState(chunk_id=chunk.chunk_id, content=chunk.text, history=["orchestrator: start"])
        for attempt in range(max_attempts):
            LOGGER.debug("chunk %s attempt %s", chunk.chunk_id, attempt + 1)
            state = creative.propose(state)
            state = compiler.run(state)
            state = evaluator.evaluate(state)
            if state.evaluation == "PASS":
                completed.append(state)
                break
            state.failed_attempts += 1
            state.log(f"orchestrator: failed_attempts={state.failed_attempts}")
            if state.failed_attempts >= 2:
                state = researcher.augment(state)
        else:
            state.log("orchestrator: max attempts reached")
            completed.append(state)
    return completed


def _demo_run(text: str) -> None:
    chunk = ContentChunk(chunk_id="demo-0", text=text, metadata={"page": 1})
    results = run_layout_graph([chunk])
    final = results[0]
    print("=== DEMO RESULT ===")
    print(final.candidate_latex or "<no latex>")
    print("--- history ---")
    for entry in final.history:
        print(entry)
    print(f"Status: {final.evaluation} ({final.score_notes})")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the experimental agent graph demo.")
    parser.add_argument(
        "--demo-text",
        default="Balance the momentum equation for incompressible flow.",
        help="Chunk text used for the demo run.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    _demo_run(args.demo_text)


if __name__ == "__main__":
    main()
