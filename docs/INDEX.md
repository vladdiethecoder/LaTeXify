# Documentation Index

## Orientation
- [REPO_DOCUMENTATION.md](../REPO_DOCUMENTATION.md) — master overview that maps the repo layout, mission, and hygiene contracts.
- [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) — system intent, stage-by-stage flow, and run-directory anatomy.
- [release/README.md](../release/README.md) — operational handbook for the production runner, bootstrap scripts, and stage artifacts.
- [release/AGENTS.md](../release/AGENTS.md) — MCP/agent playbook plus validation + hygiene expectations (mirrors the CLI harness rules in the root `AGENTS.md`).

## Pipeline & Data Artifacts
- [BLUEPRINT.md](BLUEPRINT.md) — ingestion + synthesis architecture deep dive.
- [MATH_PIPELINE_PLAN.md](MATH_PIPELINE_PLAN.md) — strategy for equation-heavy fidelity work.
- [DATA_PATHWAY_SCHEMA.md](DATA_PATHWAY_SCHEMA.md) — JSONL/Markdown lineage format for `build/run-*` logs so agents can replay runs deterministically.
- [TRAINING_DATA_ROUTE.md](TRAINING_DATA_ROUTE.md) — dataset provenance, manifests, and filing plan for every LaTeXify training track.

## Training & Fine-Tuning
- [FINE_TUNE_GUIDE.md](FINE_TUNE_GUIDE.md) — copy/paste recipes for LayoutLM token classifiers, table/multi-modal models, and LaTeX LLM alignment.

## Decisions & Research Notes
- [ADR: Local PDF → LaTeX Stack](adr/ADR-pdf_to_latex_stack.md) — canonical decision record for the pipeline architecture (supersedes `PROJECT_MAP.*`).
- [PDF_TO_LATEX_RESEARCH.md](PDF_TO_LATEX_RESEARCH.md) — pointer stub that forwards readers to the ADR while retaining historical context.
