# Contributing to LaTeXify

Thanks for helping improve LaTeXify! This guide explains the repository layout, workflows, and expectations for new contributions.

## 1. Getting Started
1. **Clone & install**
   ```bash
   git clone https://github.com/<org>/LaTeXify.git && cd LaTeXify
   python -m venv .venv && source .venv/bin/activate
   pip install -e '.[dev,ocr,llama]'
   ```
2. **Tooling**
   - TeX toolchain with `latexmk` + `chktex` is required for QA tests.
   - GPU-heavy agents (InternVL, Florence) expect endpoints configured via env vars (see `HOW_TO_INSTALL.md`).
3. **Run smoke tests**
   ```bash
   pytest tests/test_tex_assembler.py tests/test_qa_validator.py -q
   ```

## 2. Repository Layout
- `latexify/ingestion/` – OCR backends, chunkers, and the council orchestrator.
- `latexify/pipeline/` – Planner, retrieval, specialist router, synthesis, QA, aggregator, critic.
- `latexify/assembly/` – Fallback snippet generators, text/table helpers.
- `configs/` – Hydra configs (e.g., `router.yaml`).
- `docs/` – `SCHEMAS.md`, developer notes, prompt specs under `docs/tasks/`.
- `kb/` – Knowledge-base inputs + build logs.
- `tests/` – Pytest suites per subsystem.

## 3. Adding New Components
### Ingestion Backends
1. Implement a `BaseCouncilBackend` subclass under `latexify/ingestion/backends/` with an async `process` method.
2. Register it in `latexify/ingestion/ingest_pdf.py::_build_backends` (behind a feature flag or env var if it has external deps).
3. Emit metadata describing confidence/limitations; validator + QA reporting depends on it.
4. Add or extend tests under `tests/test_ingestion_*` to cover the new backend’s classification/merge logic.

### Specialist Routers / Handlers
1. Update `latexify/pipeline/specialist_router.py` to expose new tags or routing rules.
2. Implement the specialist logic (e.g., new table/figure/maths handler) inside `latexify/pipeline/synth_*` or under `latexify/assembly/`.
3. Document new tags/weights in `configs/router.yaml` and describe usage in `docs/SCHEMAS.md` if payloads change.
4. Provide pytest coverage in `tests/test_specialist_router.py` or synth-specific suites.

## 4. Running Tests & QA
- **Full suite:** `pytest -q` (may take several minutes because ingestion + QA tests spin up fake assets).
- **Selected tests:** use the per-module files under `tests/` for faster iteration.
- **Static QA:** `python -m latexify.pipeline.qa_validator ...` ensures `chktex`/critic fixes behave.
- **Docs & type checks:** run `ruff check .` and `mypy latexify` when modifying shared utils.

## 5. Style Guidelines
- Follow PEP 8 + type hints; prefer dataclasses for structured payloads.
- Keep logging structured via `latexify.utils.logging` (avoid bare `print`).
- Maintain ASCII-only source unless interacting with TeX payloads.
- All plan/consensus/layout changes must stay schema-compliant (`docs/SCHEMAS.md`). Update schemas & docs when you add fields.
- Configuration knobs belong in `configs/` and should be surfaced via Hydra overrides rather than hard-coded constants.

## 6. Pull Requests
1. Include a brief description of the problem + solution, plus relevant CLI/test output.
2. Update documentation (README/CONTRIBUTING/docs/DEV_NOTES.md) whenever you add new stages, schemas, or configs.
3. Keep commit history clean; rebase onto main if necessary.
4. Confirm CI / local pytest runs cleanly before requesting review.

Questions? Open a GitHub issue or ping maintainers in the discussion threads. Happy hacking!
