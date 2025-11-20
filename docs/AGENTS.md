# LaTeXify – Project Context for Gemini CLI

LaTeXify is a local-first pipeline for turning messy source material
(handwritten notes, scanned PDFs, mixed digital documents) into structured,
LiX-class LaTeX “textbooks”. It uses:

- An OCR ensemble (classical OCR, vision-language models, layout parsers).
- A retrieval + synthesis layer for section-level reconstruction.
- A LaTeX synthesis engine and quality gates (vision QA, snippet fusion, etc.).
- PDF compilation plus automated checks on structure and legibility.

This file explains how to work effectively inside this repo with Gemini CLI.

---

## Repository layout (high-level mental map)

You don’t need an exhaustive map; just enough to orient yourself.

- **`run_release.py`**  
  Main orchestration script for the end-to-end pipeline and larger experiments.

- **`release/`** – Production pipeline and related assets  
  - `release/pipeline/` – Core pipeline stages and orchestration helpers.  
    - `vision_synthesis.py` – Multi-view image cropping / padding & view manifests.  
    - `vision_agents.py` – Structural / mathematical / semantic / style vision agents.  
    - `snippet_fusion.py` – Snippet scoring and fusion quality gate.  
  - `release/models/` – Model configs and wrappers (OCR ensemble, VLMs, etc.).  
  - `release/README.md` – Human-oriented overview of the release pipeline.

- **`experiments/`** – Prototypes, ablation studies, and throw-away notebooks.

- **`tests/`** (or `release/tests/` if present) – Automated tests & regression suites.

- **`docs/`** – Human-facing docs; detailed prose should live here rather than in
  this file.

If a directory has its own `GEMINI.md`, those instructions refine or override
the ones here for that subtree.

---

## Building and running

Before changing pipeline code or configs, you should be able to:

1. **Create/activate the virtualenv** (if the project uses one):

   ```bash
   # Example – adapt if the repo uses a different venv path
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
