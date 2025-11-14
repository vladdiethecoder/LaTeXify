# LaTeXify Release Pipeline

A streamlined LaTeXification runner that ingests a draft PDF, rebuilds the
structure with specialist agents, and emits a clean `main.tex` (plus optional
PDF). The top-level `release/` folder is now organized by responsibility so it is
much easier to navigate:

```
release/
├── pipeline/        # ingestion, planning, retrieval, synthesis, assembly…
├── core/            # common dataclasses, reference loaders, Unicode sanitizer
├── models/          # LLM refiner + adapters for local OCR models
├── agents/          # experimental LangGraph-style scaffolding
├── scripts/         # CLI utilities (bootstrap, PPO harness, RAG eval, ...)
├── reference_tex/   # curated style exemplars consumed by the refiner/RAG
├── inputs/ & outputs/ (sample PDFs + generated artifacts)
└── tests/           # focused unit + smoke coverage
```

## One-Command Bootstrap
1. Install Python 3.10+ and Git, then clone the repo.
2. Run `python release/scripts/bootstrap_env.py` (downloads dependencies + all
   local models: Qwen layout/judge, InternVL, Florence-2, Nougat, MinerU
   (MinerU is cloned directly from GitHub: `https://github.com/opendatalab/MinerU`).
3. Install Python requirements via `python release/scripts/install_dependencies.py`
   (installs `release/requirements.txt`, PyMuPDF, and any `--extra` packages you pass).
4. Activate `release/.venv` and run the pipeline:

```bash
source release/.venv/bin/activate
python run_release.py --pdf release/samples/sample.pdf --skip-compile
```

The bootstrapper accepts useful overrides: `--venv`, `--skip-models`,
`--force-models`, or `--models layout/qwen2.5-vl-32b ocr/nougat-small ...`.
Visiting `huggingface-cli login` first dramatically reduces installation time.

## Pipeline Stages
Each stage is a function under `release/pipeline/` and appears exactly once in
`run_release.py`. The table below lists the key actors and their outputs:

| Stage (module)                 | Responsibility                                         | Output file(s)              |
| ------------------------------ | ------------------------------------------------------ | --------------------------- |
| `ingestion.py`                 | PDF text/OCR fusion, page manifests, chunk metadata    | `artifacts/chunks.json`     |
| `planner.py`                   | Section layout + master plan JSON                      | `master_plan.json`          |
| `layout.py` / `structure_graph.py` | Detect-order-construct graph + block ordering     | `plan.json`, `graph.json`   |
| `retrieval.py` + `rag.py`      | Build per-run exemplar index                           | `retrieval.json`, RAG cache |
| `synthesis.py` + `specialists.py` | Dispatch block specialists & register packages    | `snippets.json`, `preamble.json` |
| `models/llm_refiner.py`        | Optional Qwen refiner w/ style exemplars               | refined snippets in-memory  |
| `assembly.py` + `core/sanitizer/` | Compose `_preamble + body + postamble`, sanitize | `outputs/<run>/main.tex`    |
| `validation.py`, `critique.py`, `metrics.py`, `reward.py` | Compile & score results | `validation.json`, `metrics.json`, `rewards.json` |

All stages use the shared dataclasses from `release/core/common.py`, and the new
sanitizer guarantees Unicode-safe TeX output by default (disable via
`--no-unicode-sanitizer`).

## Layout-Aware Ingestion & OCR
- **Hybrid segmentation.** `ingestion.py` now leans on PyMuPDF to build a
  structured `LayoutRegion` graph per page (columns, headings, figures,
  questions, answers). The semantic chunker consumes those hints to prevent the
  entire page from collapsing into a single “figure” block and to keep question
  statements separate from their answer workspaces.
- **Specialized math OCR.** The OCR fallback stack adds a math-tuned
  `MathVisionAdapter` (Hugging Face `AutoModelForVision2Seq`, e.g.
  `microsoft/trocr-base-handwritten`) alongside Florence-2, InternVL, Nougat and
  pytesseract. Chunk metadata records the chosen backend so downstream agents
  can reroute equations to the right specialist.
- **Optional CLIP verifier.** Setting
  `LATEXIFY_ENABLE_CLIP_VERIFIER=1` reorders OCR transcripts per page using a
  lightweight CLIP similarity score between the rasterized page and each text
  candidate, which dramatically reduces the “verbatim prompt in output” issue
  on multi-modal worksheets.
- **Figure de-duplication + metadata hygiene.** The chunker fingerprints the
  `includegraphics` assets and filters duplicates before they reach assembly.
  Template prompt text is scrubbed by the sanitizer and the post-assembly
  quality check now fails the run if any exemplar instructions leak into the
  final `main.tex`.

## Streaming Planner UI & Mock LangGraph Harness
- The repository root ships a planner-first streaming prototype under
  `backend/` and `apps/ui/`. `make dev` launches a FastAPI SSE backend and the
  Streamlit “holographic builder” that renders the planner’s JSON layout plan in
  ~2 seconds, streams block tokens into column placeholders, and mirrors agent
  status updates in the sidebar.
- `curl -N http://localhost:8000/events` produces newline‑delimited SSE payloads
  (`plan → status → token → block_done → done`) that satisfy
  `schemas/events.schema.json` / `tests/test_event_contract.py`.
- Everything runs in mock mode today, but the skeleton `backend/app/graphs/*`
  is wired for LangGraph once real planner/specialist nodes are ready.

### Supporting Modules & Utilities
- `release/pipeline/semantic_chunking.py` — loads a local
  `sentence-transformers` encoder (or the hashing fallback) to cut chunks at
  semantic breakpoints; controlled via `LATEXIFY_SEMANTIC_CHUNKER_*` env vars.
- `release/pipeline/math_classifier.py` — rule-first (optional transformer) math
  classifier used by ingestion to tag each chunk with roles such as
  `display-equation`, `proof-step`, or `function-definition` for downstream
  templating.
- `release/pipeline/ingestion.py` → `DocumentStructureAnalyzer` — combines a
  PyMuPDF layout pass, heuristics, and optional LayoutLMv3 refinement to produce
  per-page document trees (headers, questions, tables, figures) consumed by the
  semantic chunker and synthesis stages.
- `release/pipeline/structure_graph.py` — turns `plan.json` + `chunks.json`
  into a Detect-Order-Construct style graph with figure/image references,
  powering downstream validation and visualization.
- `release/pipeline/synthesis_coverage.py` — compares `master_plan.json` to
  `snippets.json` and emits a concise gap report (`missing_chunk_ids`) that the
  runner writes beside each output directory.
- `release/pipeline/reward_mm.py` — renders the compiled PDF’s first page and
  feeds it through the local InternVL checkpoint to score layout aesthetics.
- `release/models/math_ocr.py` — math-aware OCR ensemble (YOLO detector +
  pix2tex recognizer + syntax validator) exposed via the new `mathocr`
  backend for `ingestion.OCRFallback`.
- `release/core/reference_loader.py` — enumerates curated exemplar `.tex`
  snippets with domain tags so retrieval + the LLM refiner share the same view
  of available styles.
- `release/core/sanitizer/` — layered Unicode + macro sanitizers invoked by
  `assembly.py` to keep PDFs compilable even when OCR drifts.
- **Math pipeline roadmap:** see `docs/MATH_PIPELINE_PLAN.md` (repo root) for the
  15-point, all-local upgrade plan covering math OCR ensembles, layout
  understanding, template-driven generation, error repair, and iterative
  refinement.

## Models & Caches
- **LLM Refiner** (`release/models/llm_refiner.py`)
  - Default repo: `Qwen/Qwen2.5-1.5B-Instruct`; overridden by
    `LATEXIFY_LLM_REPO` or `--llm-repo`.
  - Style exemplars loaded from `release/reference_tex/<domain>/` using
    `core/reference_loader.py`.
  - Honours `HF_HOME`, `TRANSFORMERS_CACHE`, and `LATEXIFY_LLM_DIR` for weight
    storage.
- **OCR stack** (`release/pipeline/ingestion.py` + `release/models/model_adapters.py`)
  - Florence‑2 ➜ InternVL ➜ Nougat ➜ Tesseract run for every page when available.
  - Missing checkpoints auto-install (thanks to the bootstrapper) and the code
    clearly warns when optional wheels (`einops`, `timm`) are required.
  - Provenance for *all* OCR transcripts is stored on each chunk via
    `metadata["ocr_backends"]`.
  - A math-specific TrOCR backend (`--ocr-mode mathvision`) loads a
    VisionEncoderDecoder checkpoint fine-tuned on handwritten formulas for
    higher fidelity and automatically cascades after Nougat when running in
    `--ocr-mode auto`.
- **RAG cache** (`release/pipeline/rag.py`)
  - `rag.build_index(reference_tex/, cache_path)` parses curated `.tex` files into
    environment-level exemplars with type + domain tags.
  - Use `--rag-refresh` or `python release/scripts/rag_eval.py` to keep caches
    current.
- **Multimodal Reward** (`release/pipeline/reward_mm.py`)
  - Uses the local InternVL-3.5 checkpoint to judge the rendered PDF directly.
  - Requires Poppler + `pdf2image` so the first page can be rasterized and scored.
  - Override the checkpoint directory with `LATEXIFY_MM_REWARD_MODEL_DIR` if your
    models live outside `release/models/ocr/`.

## Testing & Tooling
Common dev commands (run from repo root with `PYTHONPATH=.`):

```bash
# Targeted unit tests
pytest release/tests/test_ingestion_ocr_merge.py

# Full smoke (slow, downloads OCR/LLMs as needed)
pytest release/tests/test_smoke_release.py -m "not slow"

# Formatting / linting (if ruff is installed)
ruff check release && ruff format --check release
```

See `release/tests/` for focused coverage on ingestion, RAG, rewards, unicode
sanitization, and pipeline smoke tests.
After each run the pipeline performs post-build QA (template token scan, duplicate figure detection, `latexmk` compile). Failures surface as `QualityCheck` entries in `agent_metrics.json` and the run exits non-zero so CI catches regressions early.

## Command-Line Flags (run_release.py)
- `--pdf`, `--title`, `--author`, `--run-dir`
- `--chunk-chars` (ingestion chunk size), `--skip-compile`, `--no-unicode-sanitizer`
- `--rag-cache`, `--rag-refresh`
- `--reward-mode {heuristic, mm}`
- `--llm-mode {auto, off}`, `--llm-repo`, `--style-domain`
- Semantic chunker controls: `--chunker-distance-threshold`, `--chunker-min-sentences`,
  `--chunker-backend`, `--chunker-encoder-name`, plus policy knobs
  `--chunker-download {auto,always,never}` and `--chunker-hash-fallback {auto,allow,disable}`
  for reproducible planning experiments.
- Benchmarking: `--benchmark-dir` + `--benchmark-limit` runs multiple PDFs back
  to back and collates metrics under `release/outputs/benchmark_summary.json`.

## Extensibility (Agents & Scripts)
- The `release/agents/` package now houses the experimental LangGraph-style
  orchestrator (creative ➜ compile + repair ➜ evaluator ➜ research escalation).
  The ResearchAgent now uses DuckDuckGo (via `duckduckgo-search`) by default,
  but you can force offline mode with `RESEARCH_AGENT_OFFLINE=1`.
- `release/scripts/run_ppo_syntax.py` + `release/scripts/pdf_render.py` provide a
  PPO harness that renders policy responses through curated LaTeX templates and
  feeds them back into the pipeline for reward shaping.
- `release/scripts/bootstrap_env.py` is the recommended entry point for fresh
  environments; it installs Python deps, downloads every required model (layout,
  judge, OCR, MinerU), and prints activation instructions.
- `release/scripts/install_dependencies.py` is the follow-up "one click"
  installer that runs `pip install -r release/requirements.txt` plus PyMuPDF
  and any extras you pass with `--extra package==version`.

## Tips
- Set `HF_HOME` to a shared SSD so all downloads (LLM refiner + OCR models) are
  reused across runs.
- Use `LATEXIFY_SEMANTIC_CHUNKER_ALLOW_DOWNLOAD=1` to let the semantic chunker
  pull `sentence-transformers/all-MiniLM-L6-v2` on demand, or pin the files in
  your offline cache ahead of time.
- Heavy OCR backends (Florence-2 ≈10 GB, InternVL ≈22 GB) reserve GPU memory in
  priority order (`LATEXIFY_OCR_GPU_PREF`, default `0,1`). They will raise if no
  GPU has enough headroom unless you explicitly set
  `LATEXIFY_OCR_FORCE_HEAVY=1`. This keeps the exact same OCR models without
  silently falling back to heuristics.
- Heavy OCR models now stay loaded for the entire ingestion run (fewer reloads,
  less VRAM fragmentation). Set `LATEXIFY_OCR_RELEASE_MODE=page` if you truly
  need to unload them after every page.
- MathVision/TrOCR is CPU-friendly; enable it with `--ocr-mode mathvision` when
  transcribing formula-heavy scans or let the auto cascade invoke it between
  Nougat and Florence.
- PyMuPDF is preferred for PDF rasterization (and falls back to pdf2image if
  Poppler is installed). Installing dependencies via
  `python release/scripts/install_dependencies.py` brings PyMuPDF along.
- The LLM refiner automatically falls back to CPU whenever no GPU exposes
  ≥6 GB free memory. Override with `LATEXIFY_LLM_DEVICE` or
  `LATEXIFY_LLM_MIN_FREE_GB` if you want a different threshold/placement.
- The Qwen-based LLM refiner is now instantiated *after* ingestion and, by
  default, lands on the last available GPU so that OCR can own the first one.
  Override with `LATEXIFY_LLM_DEVICE` (e.g., `cuda:1` or `auto` to re-enable
  Accelerate sharding).
- The multimodal reward path (`--reward-mode mm`) loads InternVL locally and rates
  the compiled PDF preview. Set `LATEXIFY_MM_REWARD_MODEL_DIR` if the checkpoint
  is stored elsewhere.
- Every pipeline run now writes `checkpoint.log` under its output directory.
  Each line is a JSON event (stage start/success/error) plus process/system RAM,
  so if the terminal/window dies you can reopen the log to see where the run
  stopped.
- If `HF_HOME`/`TRANSFORMERS_CACHE` point to unwritable locations (e.g., a
  read-only external drive), the runner now falls back to
  `release/models/hf_cache/` automatically.
- Inspect per-stage run-time under `outputs/<run>/agent_metrics.json` when
  optimizing latency.
- Data quality matters: garbage-in leads to garbage-out. Encourage contributors to upload high-resolution PDFs, de-skew noisy scans, and prefer vector exports. Heavy handwriting should be routed through the math-friendly OCR backend (TrOCR) for best accuracy.
- Keep a feedback loop. Track user corrections (e.g., mark mis-converted equations or captions) and feed them back into fine-tuning datasets so the specialist agents keep improving across textbook/assignment styles.

Happy TeX-ing!

## Style & Contribution Guidelines
- Follow semantic LaTeX practices: keep line lengths short, add contextual `% comments` above complex derivations, and rely on macros/preamble commands instead of raw formatting. The repository ships macros such as `\Question{}` and `\Transform{}` to keep documents consistent.
- Prefer vector figures (PDF/SVG) over raster images and ensure caption text uses the document font size. If you must include bitmaps, crop them tightly and avoid duplicate inserts— the assembly step deduplicates assets automatically, but clean inputs produce superior PDFs.
- When adding new pipelines or agents, update `release/README.md` plus the relevant `AGENTS.md` entry so downstream contributors know how to reproduce the setup.
