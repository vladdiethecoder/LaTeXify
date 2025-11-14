# Experimental Agent Stack

The `release/agents/` package scaffolds a LangGraph-style workflow that can be
plugged into the release pipeline once the LLM refiner is the default path.

## GraphState
- `chunk_id` / `content` – the textual payload (page/chunk) under refinement.
- `candidate_latex` – current draft.
- `failed_attempts` – drives routing decisions (ResearchAgent after ≥2 fails).
- `evaluation` / `score_notes` – result from the EvaluatorAgent.
- `research_snippets` – external hints appended by ResearchAgent.
- `history` – trace of agent hops for debugging or LangGraph visualization.

## Agent Roles
1. **CreativeAgent** – drafts LaTeX directly from the chunk and available
   research hints (future: replace with LLM node).
2. **CompileAndRepairAgent** – simulates the compile loop, adding minimal
   structural repairs when environments are unbalanced.
3. **EvaluatorAgent** – scores the candidate snippet (`PASS`/`FAIL`).
4. **ResearchAgent** – ships with a zero-cost DuckDuckGo backend (via
   `duckduckgo-search`) so it can retrieve live snippets without API keys.
   When networking is unavailable, set `RESEARCH_AGENT_OFFLINE=1` to fall back
   to the deterministic stub used in CI.

## Orchestrator
`release/agents/orchestrator_graph.py` exposes `run_layout_graph(...)` that
executes Creative→Compile→Evaluate, automatically escalating to ResearchAgent
after repeated failures. It is intentionally LangGraph-ready: GraphState is the
edge payload, and routing logic is centralized in a single pure-Python runner.

Run a quick demo:

```bash
python -m release.agents.orchestrator_graph --demo-text "Balance the momentum equation."
```

The resulting history shows how many attempts were required, whether research
hints were injected, and the final evaluation outcome. Use this scaffold to wire
in real chunk data and LangGraph nodes once the research backend is available.
