# LaTeXify Repository Overview

This repository builds a fully local PDF ➜ LaTeX transformation stack that converts noisy worksheets, textbooks, and research notes into semantic, editable TeX. The codebase is split between (a) the production-grade `release/` runner that ingests PDFs and emits `main.tex`/PDF, (b) a streaming planner UI backed by FastAPI/Streamlit prototypes, and (c) research scaffolding for LangGraph-style agents and math-specific upgrades. This document summarizes the intent and major components so both humans and LLM copilots can reason about the system without spelunking through the tree.

## Mission & Product Goals
- **Textbook-quality LaTeX.** Output should mirror professional typesetting—structured chapters/sections, reusable `question`/`answer`/`table` templates, and properly escaped math—not bitmap screenshots of the source PDF.
- **Deterministic, local pipeline.** Every critical step (OCR, LLM refinement, retrieval, reward models) runs with locally cached checkpoints via `release/scripts/bootstrap_env.py`, making the stack reproducible on air‑gapped machines.
- **Layout-aware understanding.** LayoutLM/Florence-derived analyzers, semantic chunking, and structure graphs ensure the pipeline manipulates semantically typed blocks instead of raw page text.
- **Self-healing QA.** Each run logs stage checkpoints, validates Unicode/LaTeX safety, compiles with `latexmk`, scores the output, and can escalate to repair agents when snippets fail.
- **Human-in-the-loop tooling.** The backend + Streamlit prototype stream the planner plan/token events so designers can visualize how agents interpret a document before TeX is emitted.

## Repo Layout at a Glance
- `release/` – Production runner: pipeline stages, common dataclasses, Unicode sanitizer, OCR/LLM adapters, reference TeX exemplars, scripts, focused tests, and sample inputs/outputs.
- `run_release.py` – Orchestrates ingestion ➜ assembly ➜ critique ➜ reward, enforces Hugging Face cache locations, and exposes CLI flags for PDF selection, chunking policies, RAG refresh, benchmarking, etc.
- `apps/ui/` – Streamlit/Gradio front-ends that consume the backend SSE stream to display planner states and agent tokens.
- `backend/` – FastAPI SSE service with LangGraph-ready adapters; `GET /events` mirrors `schemas/events.schema.json` and powers the UI + `tests/test_event_contract.py`.
- `docs/` – Architectural blueprint, agent notes, and the math pipeline roadmap that guide longer-term direction.
- `schemas/`, `scripts/`, `models/`, `tests/`, `release/reference_tex/` – Shared schema contracts, utility CLIs, local checkpoints, and regression tests for ingestion/RAG/sanitizer/reward subsystems.

## Documentation Surfaces
Treat these files as the canonical breadcrumb trail when updating code or onboarding collaborators:

- [`docs/INDEX.md`](docs/INDEX.md) – living sitemap that links every spec/guide (ADR, overview, math plan, training data guide, fine-tune recipes).
- [`docs/PROJECT_OVERVIEW.md`](docs/PROJECT_OVERVIEW.md) – high-level intent, stage breakdowns, and run-directory layout; read before asking an agent to change pipeline behavior.
- [`docs/DATA_PATHWAY_SCHEMA.md`](docs/DATA_PATHWAY_SCHEMA.md) – defines the JSONL + Markdown lineage logs dropped into `build/run-*` so humans/agents can replay any run deterministically.
- [`docs/TRAINING_DATA_ROUTE.md`](docs/TRAINING_DATA_ROUTE.md) – single source of truth for staging datasets, manifests, and track assignments (`T1`–`T8`).
- [`docs/FINE_TUNE_GUIDE.md`](docs/FINE_TUNE_GUIDE.md) – copy/paste-ready commands for LayoutLM/token-classifier fine-tunes, PPO/DPO reward runs, and LaTeX model alignment.
- [`docs/adr/ADR-pdf_to_latex_stack.md`](docs/adr/ADR-pdf_to_latex_stack.md) – ADR capturing the PDF ➜ LaTeX product decision (supersedes the legacy `PROJECT_MAP.*` artifacts).
- [`release/README.md`](release/README.md) – operational guide for the production runner, bootstrap scripts, and stage-specific expectations.
- [`release/AGENTS.md`](release/AGENTS.md) – MCP playbook for LangGraph-style agents plus repo hygiene requirements (update it whenever agent contracts or schema keys move).
- [`REPO_DOCUMENTATION.md`](REPO_DOCUMENTATION.md) – this overview; keep it synchronized with the documents above so copilots always have a stable context package.

## End-to-End Release Pipeline
Each stage lives under `release/pipeline/` and appears exactly once inside `run_release.py`. Typical flow for `python run_release.py --pdf release/samples/sample.pdf --skip-compile`:
1. **Bootstrap & caches.** `release/scripts/bootstrap_env.py`/`install_dependencies.py` create `release/.venv`, fetch Florence/InternVL/Nougat/Qwen/etc., and configure Hugging Face caches inside `release/models/hf_cache` when system paths are unwritable.
2. **Ingestion (`ingestion.py`).** Builds page manifests via PyMuPDF, merges OCR transcripts (Florence ➜ InternVL ➜ Nougat ➜ Tesseract ➜ optional MathVision/TrOCR), fingerprints figures, and captures per-chunk metadata (regions, OCR provenance, layout hints).
3. **Planning (`planner.py`, `layout.py`, `structure_graph.py`).** Produces a master structural plan, optionally enriches it with detect-order-construct graphs, and emits JSON descriptions that downstream stages treat as source-of-truth.
4. **Retrieval/RAG (`retrieval.py`, `rag.py`, `release/reference_tex`).** Indexes curated exemplar snippets by domain/style so each block can request stylistic guidance.
5. **Synthesis (`synthesis.py`, `specialists.py`, `models/llm_refiner.py`).** Routes semantic block types to dedicated agents (text/math/table/figure). The optional Qwen-based refiner pulls exemplar contexts and rewrites snippets into textbook-quality TeX.
6. **Assembly & sanitization (`assembly.py`, `core/sanitizer`).** Stitches `_preamble`, snippet body, and postamble, deduplicates figures, normalizes Unicode, injects template macros, and writes `build/runs/<run>/main.tex`.
7. **Validation & scoring (`validation.py`, `critique.py`, `metrics.py`, `reward.py`, `reward_mm.py`).** Compiles LaTeX, checks for template leaks/duplicate figures, emits metrics JSON, and (optionally) scores PDF prettiness via InternVL.
8. **Coverage tracking (`synthesis_coverage.py`, `iterative_refiner.py`).** Verifies every plan chunk produced a snippet, surfaces gaps, and can re-run weak sections in a structured repair loop.
9. **Visual regression & active learning (`visual_regression.py`, `active_learning.py`).** Rasterizes the compiled PDF for SSIM-like diffs against cached page renders and writes an active-learning queue (`active_learning_queue.jsonl`) so the weakest chunks can be triaged or fed back into training.

Every stage logs to `checkpoint.log` with memory stats, so aborted runs can be resumed or debugged. Environment switches—`LATEXIFY_SEMANTIC_CHUNKER_*`, `LATEXIFY_LLM_REPO`, `LATEXIFY_OCR_FORCE_HEAVY`, `LATEXIFY_ENABLE_CLIP_VERIFIER`, etc.—make behaviors explicit for automation.

## Supporting Subsystems & Research Work
- **Semantic chunking & math classification.** `release/pipeline/semantic_chunking.py` and `math_classifier.py` cut pages on true semantic boundaries and tag chunks as display equations, proofs, tables, or instructions to drive template selection.
- **Reference loaders & sanitizer.** `release/core/reference_loader.py` enumerates exemplar `.tex` assets with domain tags; `release/core/sanitizer/` guarantees Unicode-safe output by escaping legacy OCR artifacts before compilation.
- **Scripts & tests.** `release/scripts/run_ppo_syntax.py`, `rag_eval.py`, `pdf_render.py`, and `bootstrap_env.py` cover reproducible bootstrap, PPO reward shaping, RAG maintenance, and PDF rendering. `release/tests/` contains deterministic tests for ingestion/OCR merging, RAG correctness, sanitizer behavior, and smoke coverage.
- **Experimental LangGraph agents.** Under `release/agents/`, `CreativeAgent`, `CompileAndRepairAgent`, `EvaluatorAgent`, and `ResearchAgent` cooperatively refine snippets while tracking graph state (`candidate_latex`, `failed_attempts`, `research_snippets`). `python -m release.agents.orchestrator_graph --demo-text ...` illustrates the Creative➜Compile➜Evaluate➜Research escalation path.
- **Streaming planner experience.** `backend/app` exposes SSE endpoints that stream `plan`, `status`, `token`, `block_done`, and `done` payloads to the UI, letting designers inspect the layout plan and playback snippet generation in real time. The adapters keep parity with `schemas/events.schema.json` and the mock LangGraph harness under `backend/app/graphs/`.
- **Vision synthesis + agents.** `release/pipeline/vision_synthesis.py` now runs immediately after retrieval and writes normalized multi-view crops into `artifacts/vision_views/` plus `metadata["vision_views"]`. `release/pipeline/vision_agents.py` consumes those cached views right away, records `reports/vision_diagnostics.json`, and annotates each chunk with `metadata["vision_scores"]` / `metadata["vision_notes"]` so synthesis, snippet fusion, and QA logic can branch on visual confidence. Toggle the stack via `--enable-vision-synthesis` / `--disable-vision-synthesis` (and `LATEXIFY_VISION_SYNTHESIS_*` env overrides) so CI can run balanced/fast/quality presets; the selected preset plus branch coverage statistics now flow into `DATA_PATHWAY.*`, `quality_gate.json`, and `agent_metrics.json`.
- **Snippet fusion.** `release/pipeline/snippet_fusion.py` scores candidate snippets (specialist vs. fallback) and writes `reports/snippet_fusion.json`. The average score feeds directly into `quality_gate.json`, giving CI an early warning when snippet confidence drops.
- **Fusion engine + cross-validation.** `release/pipeline/fusion_engine.py` encapsulates reusable fusion primitives (rule-based, confidence-weighted, LLM-based, adaptive) that the CLI exposes via `--fusion-strategy`. `release/pipeline/cross_validation.py` computes branch consistency, content preservation, and layout fidelity metrics that now feed reward weighting, snippet fusion routing, and telemetry.

## Medium-Term Roadmap & Design References
- **Blueprint (`docs/BLUEPRINT.md`).** Captures the original post-mortem (avoid bitmap fallbacks, fix document understanding) and defines the textbook baseline: `book` class, semantic `tcolorbox` question/answer blocks, template-driven generation, and a compile/repair loop.
- **Math-focused plan (`docs/MATH_PIPELINE_PLAN.md`).** Details a 15-point upgrade path: YOLOv8 + pix2tex ensembles, LayoutLMv3 document trees, semantic chunking, Jinja2 template engine, constrained local LLM inference, symbolic validation, progressive assembly, iterative refinement, and resource requirements.
- **Agents overview (`release/AGENTS.md`).** Explains GraphState contract, MCP workflow, and how ResearchAgent integrates DuckDuckGo/offline stubs for CI reproducibility.
- **Doc index & ADRs (`docs/INDEX.md`, `docs/adr/`).** Replace the retired `PROJECT_MAP.*` set with curated links plus decision logs, ensuring every architectural change has a signed ADR and a discoverable pointer.
- **Data lineage (`docs/DATA_PATHWAY_SCHEMA.md`).** Documents the append-only JSONL/Markdown logs that now accompany every run under `build/run-*`.
- **Training data + fine-tune guides (`docs/TRAINING_DATA_ROUTE.md`, `docs/FINE_TUNE_GUIDE.md`).** Define how datasets map to LaTeXify tracks, where to store raw/processed assets, and which scripts to run for LayoutLM / refiner tuning.

## How to Engage with the Repository
- Run `python release/scripts/bootstrap_env.py` followed by `python release/scripts/install_dependencies.py` to hydrate the `.venv`, download OCR/LLM weights (Florence-2, InternVL, Nougat, MathVision, Qwen, MinerU), and set cache defaults.
- Execute `python run_release.py --pdf <path> --skip-compile` for a fast dry run or omit `--skip-compile` for full validation. Benchmark multiple PDFs with `--benchmark-dir`/`--benchmark-limit`.
- Visualize planner output via `make dev` (FastAPI backend + Streamlit UI), or hit `curl -N http://localhost:8000/events` to inspect raw SSE streams that satisfy `schemas/events.schema.json`.
- Modify/extend templates, agents, or OCR backends by updating `release/pipeline/*`, `release/core/*`, and the reference `.tex` library—tests under `release/tests/` ensure regressions are caught early.
- Run `python scripts/visual_regression_ci.py --run-dir build/runs/<run_id>` inside CI to fail fast when the rendered PDF drifts from the cached page rasters.
- Periodically consolidate candidate snippets via `python scripts/active_learning_loop.py --runs-root build/runs --output training_data/active_learning/queue.jsonl` so annotators can triage the weakest chunks.

## Repository Hygiene & Versioning
- Never commit generated artifacts (`build/`, `build/run-*`, `release/outputs/`, `release/artifacts/`, `release/smoke_run/`, `release/cache/`, `test-results/`, `training_runs/`, `training_data/raw/`, `training_data/processed/`, `node_modules/`, Hugging Face weight caches). These paths now live in `.gitignore`; add new entries the moment you introduce another cache or run-product.
- `.gitattributes` enforces LF line endings for scripts/TeX/JSON while marking large weight formats (`*.pt`, `*.safetensors`, `*.bin`, `*.onnx`, etc.) as binary. Keep that file updated so git never mangles future checkpoints.
- Store per-run lineage artifacts under `build/run-*` (see `docs/DATA_PATHWAY_SCHEMA.md`) and reference them instead of pasting logs into docs/PRs.
- Whenever you edit agent contracts, schema keys, or CLI flags, touch both `release/README.md` and `release/AGENTS.md` plus the doc index so instructions stay in lockstep with the code.

By grounding the entire system around typed plans, specialized agents, deterministic templates, and rich telemetry, LaTeXify turns arbitrary PDFs into searchable, editable, and pedagogically structured LaTeX while leaving hooks for LangGraph orchestration and math-heavy research extensions. Use this document as the top-level mental map before diving into specific modules.
