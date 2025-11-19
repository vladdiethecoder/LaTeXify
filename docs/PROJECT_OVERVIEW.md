# Project Overview

## Goal
LaTeXify ingests noisy, layout-rich PDFs (worksheets, textbooks, research notes)
and emits high-quality LaTeX (`main.tex`) plus a compiled PDF (`main.pdf`). The
pipeline keeps every stage local: OCR ensembles (Florence, InternVL, pix2tex),
LayoutLM-based structure analysis, a retrieval layer for style exemplars, and
LaTeX specialists/LLM refiners that polish the final document into
textbook-grade quality.
The latest release layers in progressive input-quality scoring, domain
classifiers, semantic enrichment, and a consolidated quality gate so every run
adapts to the source document instead of applying a one-size-fits-all strategy.

The output is deterministic and transparent so an engineer—or an LLM copilot—
can inspect intermediate artifacts, re-run stages, and trace how each snippet
was produced.

## High-Level Pipeline
1. **Input Ingestion & OCR** – Merge PyMuPDF text, Florence/InternVL OCR, math
   detectors, and page rasters into chunked transcripts.
2. **Quality Profiling** – Score preview + OCR outputs with
   `quality_assessor.py` to decide whether aggressive cleanup or conservative
   processing is appropriate.
3. **Planning & Layout** – Build a semantic plan (sections, questions, tables)
   and a structural graph that annotates relationships between blocks.
4. **Retrieval & Context Prep** – Pull curated reference LaTeX snippets and
   exemplars based on content domain.
5. **Domain Detection & Enrichment** – Classify the document domain, add
   theorem/proof/algorithm environments, and request domain-specific packages.
6. **LaTeX Synthesis** – Route each chunk through template specialists or the
   optional LLM refiner, producing `snippets.json` and `preamble.json`.
7. **Assembly & Compilation** – Assemble snippets into `main.tex`, sanitize
   Unicode, and run `latexmk` (unless `--skip-compile`) to generate `main.pdf`.
8. **Validation & Rewarding** – Run style/quality checks, compute metrics, and
   log rewards for downstream RL/re-ranking signals.
9. **Visual QA & Active Learning** – Compare the compiled PDF against cached
   page renders, record visual diffs, and emit an active-learning queue so
   humans (or finetuning jobs) can revisit weak snippets.
10. **Quality Gate** – Combine hallucination, compilation, and visual diffs into
    a single report for CI/publishing decisions.

### Render-Aware (optional)
- **Constraint Map Builder → Flux Renderer** – When `--emit-constraint-maps`
  is active, `master_ocr_items.json` is converted into deterministic canvases
  plus binary masks under `artifacts/rendered_pages/`. Enabling
  `--enable-render-aware` feeds those canvases into the Flux-based diffusion
  wrapper, generating photorealistic page renders that respect the rendered
  formulas and routed figures. Outputs are consolidated in
  `reports/render_aware.json` for agents/QA tooling.

Each stage drops artifacts into `build/runs/<run_id>/`, making it easy to
debug or replay.

### Streaming Backend & UI
- `backend/app` hosts a FastAPI streaming endpoint (`/events`) that surfaces the
  planner output plus stage/status updates via Server Sent Events. The dispatcher
  now calls the same `release.pipeline.planner.build_master_plan` helper used by
  `run_release.py`, so the demo UI reflects real chunk/section metadata even
  before a PDF upload flow is wired.
- `apps/ui/gradio_app.py` consumes the stream and visualizes incremental tokens
  for each plan block. Point `BACKEND` at your FastAPI host and the client will
  display plan + status updates without needing to read the run directory.

## Run Directory Layout
- `build/runs/<run_id>/main.tex` – final LaTeX.
- `build/runs/<run_id>/main.pdf` – compiled PDF (when compilation enabled).
- `build/runs/<run_id>/artifacts/` – ingestion outputs (chunks, raster cache).
- `build/runs/<run_id>/reports/*.json` – planner graph, snippets, metrics, rewards,
  agent metrics, reward traces, etc.
- Progressive artifacts live under `reports/`: `input_quality.json`,
  `domain_profile.json`, `semantic_enrichment.json`, and `quality_gate.json`.
- `build/runs/<run_id>/reports/snippet_fusion.json` – per-chunk scoring from the
  new fusion stage (also embedded into `quality_gate.json`).
- `build/runs/<run_id>/logs/checkpoint.log` – stage-by-stage telemetry.
  etc.
- `build/run-<run_id>/DATA_PATHWAY.llm.jsonl` – per-stage JSONL log.
- `build/run-<run_id>/DATA_PATHWAY.md` – human-readable summary.

The `<run_id>` suffix ties `build/runs` to `build/run-*` so you can jump
from an artifact to the corresponding data pathway logs.

## Working with LLMs
External copilots (Gemini, ChatGPT) should follow this order:
1. Read this overview (`docs/PROJECT_OVERVIEW.md`) for context about the goal,
   stages, and artifact layout.
2. Open `docs/DATA_PATHWAY_SCHEMA.md` to understand the logging format.
3. For a specific run, inspect `build/run-<run_id>/DATA_PATHWAY.md` for the
   summary, then stream `DATA_PATHWAY.llm.jsonl` to reconstruct every stage,
   including model usage and artifact paths. This makes it deterministic and
   token-efficient for LLMs to explain how any `main.tex` / `main.pdf` was
   created.
