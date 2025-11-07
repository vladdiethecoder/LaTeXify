# HOW_TO_RUN

This repository is fully self-contained: all OCR, planning, judging, and LaTeX synthesis run locally on the dual-GPU workstation. Follow the steps below to reproduce the pipeline end-to-end.

## 1. Environment Setup

1. Ensure Python 3.11+ is installed.
2. Create a virtual environment and install the package in editable mode:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -e .
   ```
3. (Optional) Populate `dev/models/` with your local GGUF / vLLM checkpoints as needed. The pipeline works offline but will call out when a model endpoint is missing.

## 2. Provide Input PDFs

- Drop PDFs into `dev/inputs/`. The folder already contains a few sample STEM worksheets.
- Use descriptive filenames; they become default document titles in the generated LaTeX.

## 3. Run the Pipeline

Use the convenience launcher `run_local.py` to tie together layout planning, ingestion, judging, TeX assembly, and QA:

```bash
python run_local.py --pdf "Basic Skills Review Unit Assessment.pdf" --title "Skills Review"
```

Key flags:

- `--pdf` – filename inside `dev/inputs/` **or** any absolute/relative path to a PDF.
- `--title`, `--author`, `--course` – metadata for the generated LaTeX front matter.
- `--langchain` – execute the orchestrator as a LangChain runnable sequence.
- `--skip-qa` – bypass the QA/auto-fix pass if you only need raw snippets.
- `--qa-compile` – attempt a latexmk build inside the QA step (requires latexmk).
- `--list-inputs` – show the PDFs found in `dev/inputs/` and exit.

All build artifacts (layout plan, consensus, snippets, QA report) land under `build/`, while model council evidence lives under `dev/runs/<timestamp>/`.

## 4. Direct CLI (Advanced)

If you prefer calling the modules yourself, the package exposes granular entry points:

```bash
latexify-layout-planner --text-file notes.txt --image scans/page1.png --out build/layout.json
latexify-ingest-pdf --pdf dev/inputs/sample.pdf --run-dir dev/runs/sample --permissive
latexify-judge-council --run-dir dev/runs/sample
latexify-tex-assembler --plan build/plan.auto.json --consensus dev/runs/sample/blocks_refined.jsonl --aggregate
```

Refer to `docs/tasks/` for the latest prompt specifications and `docs/DEV_NOTES.md` for deeper architectural notes.
