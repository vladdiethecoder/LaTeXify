# Prompt Architecture Overview (LaTeXify)

This note distills the “Prompt architecture overview” PDF into an actionable view aligned with the current LaTeXify pipeline (multi-branch orchestration, Fusion Engine, Cross-Validation, MonkeyOCR/Kimi GGUF stack).

## 1) Snapshot of Current State
- **Orchestration:** Multi-branch execution (`release/pipeline/parallel_branches.py`) with branch manifests and telemetry; orchestration still largely procedural.
- **Fusion & Validation:** Intelligent Fusion Engine and `cross_validation.py` gate branch outputs; adaptive strategies are available.
- **OCR & Vision:** MonkeyOCR default fallback; Florence/InternVL/Kimi adapters; VLM adapter registry (`release/models/vlm_adapters.py`); layout still partly heuristic.
- **Compile & Repair:** Robust compilation and repair loops exist but not yet expressed as a LangGraph graph.
- **Telemetry:** DATA_PATHWAY + branch/quality artifacts produce metrics, but coverage of math/table fidelity is partial.

## 2) Gaps Highlighted by PDF (mapped to repo reality)
- **LangGraph orchestration missing:** Planner/graph runner placeholders; loops and conditional routing not encoded. Action: express compile–repair + specialist routing as a LangGraph graph; emit graph manifest.
- **Heuristic specialists:** `release/agents/*` still contain heuristic filters. Action: parameterize with registry-based specialist selection and tie scoring to cross-validation metrics.
- **Layout brittleness / overuse of `\\includegraphics`:** Ingestion uses PyMuPDF heuristics; image fallbacks persist. Action: add detector-backed layout (LayoutLMv3/DocTron) adapter slot and minimize bitmap fallback by preferring editable reconstruction when confidence allows.
- **Fragmented model routing:** Many models exist; selection rules not centralized. Action: codify modality routing table (text/math/table/figure) using the VLM adapter registry and reuse weights across agents.
- **Quality gates limited:** Compile success and length checks dominate. Action: add math/table fidelity metrics (SymPy equivalence, table structure diffs) to quality_gate and branch summaries.

## 3) Recommended Architecture Moves (prioritized)
1. **Graphify critical loop:** Model compile–repair–validate as LangGraph nodes (ingestion → specialists → fusion → compile → cross-validate → repair). Keep state in `graph.json` for UI/debug.
2. **Specialist routing policy:** Define a routing map (text → Florence/InternVL, math → pix2tex/MathVision, tables → Table2LaTeX/TATR, figures → Qwen-VL/InternVL2) with confidence thresholds; implement in `release/pipeline/specialists.py`.
3. **Layout upgrade slot:** Introduce an optional detector-backed layout adapter; gate via runtime flag to avoid hard dependency.
4. **Neuro-symbolic checks:** Use SymPy/latex2sympy to verify equations post-OCR and before fusion; feed failures into repair loop.
5. **Template auto-selection:** Add lightweight style classifier; default Jinja2 templates per document class; fall back to current adaptive fusion when uncertain.
6. **Metrics enrichment:** Extend `quality_gate.json` / branch metrics with math/table fidelity, layout coverage, image-versus-editable share, and reroute logic based on these scores.

## 4) Concrete Tasks (short list)
- Encode LangGraph graph for compile–repair and specialist routing; persist graph manifest for visualization.
- Refine `release/pipeline/specialists.py` to centralize modality routing and reuse loaded adapters.
- Add SymPy-based equivalence check + table structure diff into cross-validation; surface in agent_metrics.
- Define template/style classifier stub and wire a config flag; keep legacy path intact.
- Add doc links and README/AGENTS pointers (this file) to keep contributors aligned.
- Expose the graph manifest to the streaming UI (`backend/app/graphs/runner.py`) so debuggers can render it.

## 5) Quick References (repo)
- Multi-branch orchestration: `release/pipeline/parallel_branches.py`, `release/pipeline/branch_outputs.py`
- Fusion & validation: `release/pipeline/fusion_engine.py`, `release/pipeline/cross_validation.py`
- OCR/VLM adapters: `release/pipeline/ingestion.py`, `release/models/vlm_adapters.py`, `release/pipeline/specialists.py`
- Compile/repair: `release/pipeline/robust_compilation.py`, `release/agents/compile_and_repair*`
- Telemetry: `release/README.md`, `release/DATA_PATHWAY...`, `quality_gate.json` artifacts

## 6) How to Use This Doc
- Treat as a living bridge between PDF critiques and the current code. When you implement an improvement, update this doc and the linked modules, and record the change in memory_ingest.
