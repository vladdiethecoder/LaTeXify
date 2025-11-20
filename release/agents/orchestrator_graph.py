"""LangGraph-ready orchestration scaffold for the tiered agent stack."""
from __future__ import annotations

import argparse
import logging
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

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


def build_graph_manifest() -> Dict[str, object]:
    """Return a lightweight LangGraph-style manifest for visualization/debugging.

    Nodes reflect the compile–repair–evaluate loop plus specialist routing entry.
    This is JSON-serializable and can be consumed by the backend/streaming UI.
    """
    nodes = [
        {"id": "specialist_router", "type": "router", "description": "Route chunk to modality specialist"},
        {"id": "creative", "type": "llm", "description": "Draft LaTeX snippet"},
        {"id": "compile", "type": "tool", "description": "pdflatex/tectonic compile"},
        {"id": "evaluate", "type": "judge", "description": "length/layout/metric checks"},
        {"id": "repair", "type": "llm", "description": "Compile-and-repair loop"},
        {"id": "research", "type": "retriever", "description": "External snippets / evidence"},
        {"id": "success", "type": "terminal", "description": "Snippet accepted"},
        {"id": "fail", "type": "terminal", "description": "Max attempts reached"},
    ]
    edges = [
        ("specialist_router", "creative", "dispatch"),
        ("creative", "compile", "draft->compile"),
        ("compile", "evaluate", "compile->evaluate"),
        ("evaluate", "success", "pass", {"condition": "evaluation == PASS"}),
        ("evaluate", "repair", "fix", {"condition": "evaluation != PASS"}),
        ("repair", "compile", "recompile"),
        ("repair", "research", "needs context", {"condition": "failed_attempts >= 2"}),
        ("research", "creative", "augment draft"),
        ("repair", "fail", "give up", {"condition": "failed_attempts >= max_attempts"}),
    ]
    return {
        "name": "latexify-compile-repair-graph",
        "version": "0.1",
        "entry": "specialist_router",
        "nodes": nodes,
        "edges": [
            {"source": src, "target": dst, "label": label, **(meta or {})}
            for src, dst, label, *rest in (edge + (None,) if len(edge) == 3 else edge for edge in edges)
            for meta in [rest[0] if rest else {}]
        ],
    }


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
    parser.add_argument(
        "--emit-manifest",
        type=Path,
        default=None,
        help="Optional path to write graph manifest JSON for UI/visualization.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    if args.emit_manifest:
        manifest = build_graph_manifest()
        args.emit_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.emit_manifest.write_text(json.dumps(manifest, indent=2))
        print(f"Wrote graph manifest to {args.emit_manifest}")
    _demo_run(args.demo_text)


if __name__ == "__main__":
    main()
