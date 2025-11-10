# Developer Notes

This document describes the current architecture, configuration surfaces, and near-term priorities for LaTeXify’s modular pipeline.

## Architecture Overview
- **Layout Planner (`latexify/pipeline/layout_planner.py`)** – Generates `layout_plan.json` (doc-class hints, section outlines, asset slots). Output schema documented in `docs/SCHEMAS.md`.
- **Ingestion Council (`latexify/ingestion/ingest_pdf.py`)** – Runs multiple OCR backends, extracts page assets, and writes consensus snippets/metadata. Assets are persisted under `build/assets/` and referenced later via plan tasks.
- **Planner Scaffold (`latexify/pipeline/planner_scaffold.py`)** – Produces `plan.json` by combining layout information, consensus blocks, and asset manifests. The resulting `PlanSchema` drives downstream agents.
- **Retrieval + Synthesis (`latexify/pipeline/retrieval_bundle.py`, `synth_latex.py`)** – Builds bundle files with OCR text plus KB matches, routes them through the specialist router (configurable via `configs/router.yaml`), and emits snippets under `build/snippets/`.
- **QA + Aggregation (`latexify/pipeline/qa_validator.py`, `aggregator.py`)** – Performs chktex + critic validation, inserts missing packages, reruns critic on failures, then assembles `main.tex` and compiles via `latexmk`.

## Configuration & Schemas
- **Hydra configs:** stored in `configs/` (currently `router.yaml`). Override at runtime using `--config-dir configs --config-name router` and pass dotted overrides for weights/tag overrides.
- **Schemas:** all cross-stage payloads are defined in `latexify/models/schemas.py` and documented in `docs/SCHEMAS.md`. Update both when adding fields to plans, consensus blocks, or layout blueprints.
- **KB / Assets:** knowledge-base artefacts live under `kb/`, and asset manifests get written during aggregation/QA. Keep path conventions stable so planner + aggregator agree on asset locations.

## Dev Workflows
1. **End-to-end smoke run:** `python scripts/pipeline.py --seed 4242` (expects TeX + chktex installed).
2. **Targeted stage debugging:** run the relevant `python -m latexify.pipeline.<stage>` module with `--help` for flags.
3. **QA iteration:** `python -m latexify.pipeline.qa_validator --plan build/plan.json --snippets build/snippets --attempt-compile` to validate snippets before aggregation.
4. **Testing:** `pytest -q` for the full suite; use `pytest tests/test_tex_assembler.py -q` or similar for faster cycles. Validator tests require `chktex`.

## Current Focus Areas
- **Synthesis Robustness:** expand table/figure specialists and keep the router weights (`configs/router.yaml`) in sync with new tags.
- **QA Feedback Loop:** continue wiring critic + auto-fix outputs into `docs/SCHEMAS.md` so downstream tooling can consume the metadata.
- **Documentation:** keep `README.md`, `CONTRIBUTING.md`, and this file aligned whenever CLI names, configs, or schemas change.
- **Evaluation:** `dev/eval` holds baseline fixtures; extend metrics when adding new document genres.

For older historical to-do lists, refer to the git history; this document now tracks the modular design only.
