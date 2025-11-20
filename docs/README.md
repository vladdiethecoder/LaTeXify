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

## Documentation & Repo Hygiene
- **Orientation.** Start with `REPO_DOCUMENTATION.md` and `docs/INDEX.md`, then dive into `docs/PROJECT_OVERVIEW.md`, `docs/DATA_PATHWAY_SCHEMA.md`, and `docs/TRAINING_DATA_ROUTE.md` before you touch pipeline code or dataset tooling. Agent/MCP requirements live in `release/AGENTS.md` (mirrored for the CLI harness in the root `AGENTS.md`).
- **Per-run lineage.** Every invocation logs `build/run-<run_id>/DATA_PATHWAY.{md,llm.jsonl}`; cite those artifacts in design docs and PRs instead of screenshotting terminals.
- **Artifact hygiene.** `.gitignore` keeps `build/`, `build/run-*`, `release/outputs/`, `release/artifacts/`, `release/smoke_run/`, `release/cache/`, `test-results/`, `training_runs/`, `training_data/raw/`, `training_data/processed/`, `node_modules/`, and `apps/ui/dist/` out of git. Add new patterns the moment you create another cache so generated PDFs/models never land in commits.
- **Line endings & weights.** `.gitattributes` normalizes LF endings for scripts/TeX/JSON and marks checkpoint formats (`*.pt`, `*.safetensors`, `*.onnx`, etc.) as binary. Extend it whenever a new format or platform-specific script shows up.
- **Training playbooks.** Follow `docs/FINE_TUNE_GUIDE.md` plus `scripts/one_click_release.py` / `scripts/register_layout_dataset.py` when refreshing LayoutLM checkpoints so the README, manifests, and CLI helpers stay in lockstep.

## One-Command Bootstrap
1. Install Python 3.10+ and Git, then clone the repo.
2. Run `python release/scripts/bootstrap_env.py` (downloads dependencies + all
   local models: DeepSeek-Coder-V2-Lite refiner, DeepSeek-V3 (for vLLM), Mixtral 8×7B,
   Qwen2.5 judges, InternVL, Florence-2, Nougat, MinerU (cloned directly from
   GitHub: `https://github.com/opendatalab/MinerU`). Pass `--verify-vllm` if you want
   the bootstrapper to spin up a tiny vLLM decode after installation.
3. Install Python requirements via `python release/scripts/install_dependencies.py`
   (installs `release/requirements.txt`, PyMuPDF, and force-installs the GPU helpers
   `einops` + `timm` so Florence-2/InternVL can stay on CUDA).
4. Activate `release/.venv` and run the pipeline:

```bash
source release/.venv/bin/activate
python run_release.py --pdf release/samples/sample.pdf --skip-compile
```

The bootstrapper accepts useful overrides: `--venv`, `--skip-models`,
`--force-models`, `--models layout/qwen2.5-vl-32b ocr/nougat-small ...`, and
`--verify-vllm`. Visiting `huggingface-cli login` first dramatically reduces
installation time. When working with the local Kimi GGUF weights, set
`LATEXIFY_KIMI_K2_VARIANT` (e.g., `Q4_K_M`, `Q3_K_M`) to download only the required
quantization and keep disk usage under control.

### One-click layout prep + release
Once you drop the real `train/val/test` JSONL splits next to the placeholder files
inside `training_data/processed/<slug>/splits/`, you can run the entire prep +
release flow via:

```bash
python scripts/one_click_release.py \
  --pdf release/samples/sample.pdf \
  --layout-model training_runs/layoutlmv3-doclaynet
```

The helper copies every `{train,val,test}.jsonl` (or `*.source.jsonl`) onto the canonical
`data.jsonl`, ensures the LayoutLM processor assets are present inside the
`training_runs/<layout>` directory, sets sane defaults for `LATEXIFY_*` env vars,
and finally invokes `run_release.py`. Pass `--skip-release` if you only want to
refresh the dataset splits, or `--force-register` to overwrite already-populated
`data.jsonl` files. If a slug is missing real `*.jsonl` exports, the helper aborts
with a per-split checklist so you know exactly which paths still need the genuine data
before the release pipeline can run. Use `scripts/register_layout_dataset.py` or copy
your exports into `training_data/processed/<slug>/splits/<split>/` (any filename other
than `data.jsonl`) and re-run the helper.

## Reference Hardware
The release runner now targets a single RTX 5090 workstation (32 GB VRAM) with
32 GB of system RAM. All defaults assume GPU `cuda:0` is that card and that
model downloads share a fast local SSD. Hosts with larger VRAM pools or
multiple GPUs can override the auto-detected behavior via
`LATEXIFY_OCR_GPU_PREF`, `LATEXIFY_OCR_RELEASE_MODE=run`, or
`LATEXIFY_LLM_DEVICE`.

## Pipeline Stages
Each stage is a function under `release/pipeline/` and appears exactly once in
`run_release.py`. The table below lists the key actors and their outputs:

| Stage (module)                 | Responsibility                                         | Output file(s)              |
| ------------------------------ | ------------------------------------------------------ | --------------------------- |
| `ingestion.py`                 | PDF text/OCR fusion, page manifests, chunk metadata    | `artifacts/chunks.json`     |
| `quality_assessor.py`          | Pre/post-OCR quality scoring, routing hints            | `reports/input_quality.json`, manifest `input_quality` |
| `planner.py`                   | Section layout + master plan JSON                      | `master_plan.json`          |
| `layout.py` / `structure_graph.py` | Detect-order-construct graph + block ordering     | `plan.json`, `graph.json`   |
| `domain_detector.py`           | Keyword + layout-driven domain profiling               | `reports/domain_profile.json` |
| `retrieval.py` + `rag.py`      | Build per-run exemplar index                           | `retrieval.json`, RAG cache |
| `parallel_branches.py`         | VRAM-aware sequencing for Branch A/B/C                 | `reports/branch_artifacts.json` |
| `branch_orchestrator.py`       | Underlying branch scheduling + checkpoints             | `reports/branch_progress.json`, `artifacts/branches/` |
| `branch_b_vision.py`           | Nougat/InternVL vision synthesis branch (Branch B)     | `reports/branch_b_vision.json`, `artifacts/branches/branch_b/` |
| `branch_outputs.py`            | Copies branch outputs + generates `sample_*.pdf`       | `branch_a/`, `branch_b/`, `branch_c/`, `reports/branch_outputs.json` |
| `vision_synthesis.py`          | Generate normalized multi-view crops for chunk metadata | metadata entry `vision_views` |
| `vision_agents.py`             | Score visual crops (structure/math/style semantics)     | `reports/vision_diagnostics.json`, metadata `vision_scores` |
| `latex_image_generator.py`     | Render aesthetic LaTeX previews per chunk               | `artifacts/latex_images/`, chunk metadata `latex_image` |
| `branch_evaluator.py`          | Compare branch quality (BLEU/visual metrics)            | `reports/branch_evaluation.json` |
| `synthesis.py` + `specialists.py` | Dispatch block specialists & register packages    | `snippets.json`, `preamble.json` |
| `semantic_enricher.py`         | Add theorem/proof/algorithm environments per domain    | `reports/semantic_enrichment.json`, updated `snippets.json` |
| `models/llm_refiner.py`        | Optional DeepSeek refiner w/ style exemplars           | refined snippets in-memory  |
| `assembly.py` + `core/sanitizer/` | Compose `_preamble + body + postamble`, sanitize | `outputs/<run>/main.tex`    |
| `hallucination.py`             | Guard section headings/snippets against hallucinations | `reports/hallucination.json` |
| `validation.py`, `critique.py`, `metrics.py`, `reward.py` | Compile & score results | `validation.json`, `metrics.json`, `rewards.json` |
| `snippet_fusion.py`            | Score snippet candidates + emit fusion confidence       | `reports/snippet_fusion.json` |
| `visual_regression.py`        | Diff compiled PDFs against cached page renders          | `reports/visual_regression.json` |
| quality gate (`hallucination.py` + `validation.py` + `visual_regression.py`) | Consolidated QA + provenance | `reports/quality_gate.json` |
| `active_learning.py`          | Collect weak/flagged chunks for human review queues     | `reports/active_learning_summary.json`, `reports/active_learning_queue.jsonl` |

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
- **Nougat adapter refresh.** `release/models/nougat_adapter.py` now extends the
  shared `BaseOCRAdapter`, batches page images, emits per-page confidence
  scores, and automatically falls back to the existing math OCR pipeline
  (pix2tex) when recognition quality drops. The ingestion step records these
  confidences inside the OCR result metadata so later stages and quality gates
  can react without additional parsing.
- **Optional CLIP verifier.** Setting
  `LATEXIFY_ENABLE_CLIP_VERIFIER=1` reorders OCR transcripts per page using a
  lightweight CLIP similarity score between the rasterized page and each text
  candidate, which dramatically reduces the “verbatim prompt in output” issue
  on multi-modal worksheets.
- **Figure de-duplication + metadata hygiene.** The chunker fingerprints the
  `includegraphics` assets and filters duplicates before they reach assembly.

## Vision Synthesis & Agents
- **VisionSynthesisConfig + MultiViewRenderer.** `vision_synthesis.py` now runs immediately after retrieval (before any text generation), reads chunk metadata (`bbox`, `page_image`, `images`), and emits padded multi-resolution crops under `artifacts/vision_views/`. Each chunk records its views inside `metadata["vision_views"]`, so synthesis/specialist/QA stages consume cached crops as a deterministic feature channel.
- **ImageViewGenerator.** Handles the heavy lifting (padding, color jitter, grayscale conversions, tensor normalization via torchvision) so downstream components can grab either PIL views or normalized tensors.
- **Vision agents.** `vision_agents.py` defines a BaseVisionAgent plus four concrete flavors (Structural, Mathematical, Semantic, Style). The suite now runs right after `vision_synthesis.py`, captions each cached view with the Florence/InternVL adapters, summarizes findings via the lightweight vLLM client, writes `reports/vision_diagnostics.json`, and annotates each chunk with `metadata["vision_scores"]` / `metadata["vision_notes"]` so synthesis, snippet fusion, and QA logic can route based on visual confidence.
- **Agent suite helper.** `VisionAgentSuite` wires the four agents together so experiments can obtain a consistent set of layout/math/style diagnostics for each chunk without touching LangGraph.
- **Runtime switches & presets.** Toggle the entire stack with `--enable-vision-synthesis` / `--disable-vision-synthesis` or the env var `LATEXIFY_VISION_SYNTHESIS_ENABLED`. Choose presets via `--vision-preset {balanced,fast,quality}` (`LATEXIFY_VISION_SYNTHESIS_PRESET`) and override internals with `LATEXIFY_VISION_SYNTHESIS_TARGET_SIZES`, `LATEXIFY_VISION_SYNTHESIS_MAX_VIEWS`, etc. The resolved config lands inside `build/run-*/DATA_PATHWAY.*` for reproducibility.
- **Aesthetic LaTeX image previews.** `latex_image_generator.py` takes the chunk metadata produced during ingestion/layout, renders styled LaTeX snippets (journal/memoir/minimal presets), and writes PNG previews into `artifacts/latex_images/`. Each chunk records `metadata["latex_image"]` and the style used, regeneration attempts are capped at three, and the generator progressively simplifies styles before falling back to a basic render or, ultimately, to the Branch A textual snapshot.

## Branch Orchestration & Vision Branch B
- **Memory-aware sequencing.** `parallel_branches.py` and `branch_orchestrator.py` hook in immediately after retrieval, walk Branch A/B/C sequentially, and snapshot system/GPU memory before deciding whether to run, defer, or skip a branch. Events are written into `DATA_PATHWAY.llm.jsonl` with a `branch` metadata payload so downstream agents know exactly which branch produced which artifacts. Progress for every branch is persisted in `reports/branch_progress.json` + `reports/branch_artifacts.json` and consolidated into `agent_metrics.json` under `BranchOrchestrator`.
- **Artifact isolation.** Every branch receives its own sandbox under `artifacts/branches/branch_<letter>/` plus a mirrored report directory. The sequencer automatically injects branch metadata and preferred resource hints (env overrides, device affinity) so branch runners do not need to duplicate boilerplate, and heavy adapters are torn down between branches to respect the 32 GB VRAM ceiling.
- **Branch B (vision synthesis).** `branch_b_vision.py` consumes the ingestion-time `vision_branch` metadata, resolves page rasters, and runs Nougat (`facebook/nougat-base` if available, otherwise Nougat-small) to produce aesthetic LaTeX snippets for complex visual regions. If Nougat fails or produces low confidence, the branch falls back to the existing pix2tex math recognizer. InternVL is reused (via the shared adapter registry) to generate layout hints that feed into the confidence scorer, and the pipeline retries generation up to `LATEXIFY_BRANCH_B_MAX_REGENERATIONS` times before declaring failure.
- **Branch outputs + samples.** `branch_outputs.py` mirrors the final LaTeX/PDF pair for every branch (`branch_a/main.tex`, `branch_b/sample_b.pdf`, etc.) and emits `reports/branch_outputs.json`. The legacy `main.tex` / `main.pdf` in `build/runs/<run_id>/` now always point to the “best” branch (currently Branch C when available) so downstream consumers remain backward compatible while still gaining access to per-branch samples `sample_a.pdf`, `sample_b.pdf`, and `sample_c.pdf`.
- **Telemetry + controls.** Confidence summaries (`avg_confidence`, `max_confidence`, failure counts) flow into `DATA_PATHWAY` and `quality_gate.json`. Use `LATEXIFY_BRANCH_B_NOUGAT_DIR`, `LATEXIFY_BRANCH_B_DEVICE`, and `LATEXIFY_BRANCH_B_DISABLE_LAYOUT` to override model directories, CUDA placement, or InternVL hints.
- **Comparative branch evaluator.** `branch_evaluator.py` consumes the orchestrator manifest, fused snippets, and LaTeX preview metrics to compute BLEU-style overlap between Branch A (raw OCR), Branch B (vision synthesis), and the fused Branch C output. The resulting `reports/branch_evaluation.json` feeds into `quality_gate.json`, `agent_metrics.json`, and the active-learning queue so low-performing branches can be routed for human review automatically.
- **Runtime controls.** Enable or disable the orchestrator with `--enable-multi-branch` (or `LATEXIFY_ENABLE_MULTI_BRANCH`), restrict execution to a subset of branches via `--branches a,c`, and cap GPU usage with `--branch-memory-limit <GB>`. Fusion strategies now accept human-friendly aliases (`--fusion-strategy confidence|rules|llm|fallback|multi_branch`) that map onto the existing rule/ensemble/LLM/adaptive engines.

## Snippet Fusion & Quality Gate
- **SnippetJudge + FusionStrategy.** `snippet_fusion.py` compares the specialist/refiner output with lightweight fallbacks (escaped OCR text) using math consistency, syntax balance, and style heuristics before choosing the best candidate or emitting a hybrid.
- **Confidence weighting.** Scoring blends math overlap, syntax balance, stylistic cues, and the upstream `snippet_confidence` note so low-trust snippets can surface in `flagged_chunks` quickly.
- **Gateway integration.** The fusion report now lands in `reports/snippet_fusion.json` and feeds into `quality_gate.json` alongside hallucination, validation, and visual regression data, giving CI a single place to inspect snippet quality regressions.
- **Intelligent Fusion Engine.** `release/pipeline/fusion_engine.py` powers the CLI-facing strategies (`select_best`, `merge_hybrid`, `ensemble_average`, `adaptive`) with four reusable engines: heuristic rule-based, confidence-weighted blending, an optional LLM-backed merger, and an adaptive router that learns from validation results so snippet fusion can react to past compile failures automatically.
- **Strategy controls.** Use `--fusion-strategy {select_best,merge_hybrid,ensemble_average}` (or `LATEXIFY_SNIPPET_FUSION_STRATEGY`) to pick the scoring policy; the choice is mirrored inside `reports/snippet_fusion.json`, `quality_gate.json`, and `agent_metrics.json`.
  Template prompt text is scrubbed by the sanitizer and the post-assembly
  quality check now fails the run if any exemplar instructions leak into the
  final `main.tex`.
- **Vision diagnostics passthrough.** `quality_gate.json` now includes both the summary from `reports/vision_diagnostics.json` and the ingestion-time branch comparison (vision signal/consistency/coverage plus the active fusion + vision presets), so CI and humans can correlate hallucination/snippet regressions with the visual cues surfaced by `VisionAgentSuite`.
- **Heavy OCR front-loading.** Florence-2 and InternVL now run before lighter
  backends whenever GPU memory allows so they can reserve VRAM while it is
  still plentiful. If VRAM is low, the pipeline logs the fallback and routes
  Florence to CPU/offload mode automatically.

## Kimi-K2 & Robust Compilation Controls
- **Runtime knobs.** `run_release.py` exposes `--kimi-temperature` and `--kimi-context-size`
  (mirrored by `LATEXIFY_KIMI_K2_TEMPERATURE` / `LATEXIFY_KIMI_K2_CONTEXT`), so every
  Kimi-backed agent (semantic validation, hallucination guardrails, latex repair)
  inherits a consistent sampling profile. The CLI updates the corresponding env
  vars before any adapters warm up, guaranteeing llama.cpp loads the requested
  context window.
- **Robust compilation toggles.** Use `--enable-robust-compilation` /
  `--disable-robust-compilation` and `--compilation-retry-count` to configure the
  incremental compiler. These flags mirror
  `LATEXIFY_ENABLE_ROBUST_COMPILATION` and `LATEXIFY_COMPILATION_RETRY_COUNT` and
  are persisted into `DATA_PATHWAY.*` so downstream agents can reason about
  recovery behavior.
- **Layout + MonkeyOCR.** `--layout-confidence-threshold` clamps low-confidence
  layout regions before chunking (default `LATEXIFY_LAYOUT_CONFIDENCE_THRESHOLD`),
  while `--enable-monkey-ocr` / `--disable-monkey-ocr` drive MonkeyOCR availability
  without editing code. Both switches flow into `ingestion.run_ingestion`, chunk
  metadata, `agent_metrics.json`, and the quality gate payload.
- **Surya layout backend.** Opt into Surya polygons with `--layout-backend surya`
  (or `LATEXIFY_LAYOUT_BACKEND=surya`). This pulls weights via
  `release/scripts/install_models.py`, requires the optional `surya-ocr`
  dependency, and emits `master_ocr_items.json` that enumerates every detected
  region (bbox + polygon + OCR routing hints). Disable Surya’s math detector with
  `--disable-surya-math-detector` if VRAM is tight or Nougat should not be
  invoked automatically.
- **Constraint maps for render-aware mode.** Enable
  `--emit-constraint-maps` (with optional `--constraint-pages 1,3`) to convert
  `master_ocr_items.json` into `rendered_pages/page_<id>_constraint.png` plus
  binary masks. This runs the new symbolic renderer (Matplotlib mathtext +
  caching) and a ReportLab compositor that places rendered formulas and figure
  crops at their absolute positions, creating deterministic canvases ready for
  diffusion-based reconstruction.
- **Flux render-aware synthesis.** Use `--enable-render-aware` to feed the
  constraint maps into a Flux.1 Fill (or compatible diffusers) pipeline that
  inpaints the remaining layout. Flags `--flux-model`, `--flux-steps`,
  `--flux-guidance`, `--flux-prompt`, `--flux-device`, and
  `--render-aware-pages` control the checkpoint, VRAM mode, and prompts. The
  stage writes `rendered_pages/page_<id>_render.png` plus
  `reports/render_aware.json` so downstream evaluators/agents can inspect the
  photorealistic outputs.
- **Telemetry.** Compilation attempt history, Kimi inference metrics, layout
  confidence stats, and cross-validation composites are logged via
  `log_stage_event` and persisted into `agent_metrics.json`,
  `reports/quality_gate.json`, and the DATA_PATHWAY schema so CI can replay
  robust-compilation and repair outcomes without parsing raw logs.

## Progressive Input Quality & Domain Routing
- **Tiered quality modes.** `quality_assessor.py` scores the document during PDF
  preview and after OCR fusion. The emitted profile
  (`reports/input_quality.json` + `manifest["input_quality"]`) labels the run as
  `low/aggressive`, `medium/balanced`, or `high/conservative`, which in turn
  reorders OCR backends, toggles aggressive math cleanup, and decides whether
  the DeepSeek refiner should rewrite or preserve a snippet.
- **Domain-aware synthesis.** `domain_detector.py` generates
  `reports/domain_profile.json` with keyword-driven confidences and recommended
  packages (e.g., `amsthm`, `algorithm2e`, `siunitx`). Assembly merges those
  packages with the PreambleAgent output so the final TeX automatically loads
  theorem/proof/algorithm helpers.
- **Semantic enrichment.** After template specialists run,
  `semantic_enricher.py` augments `snippets.json` with theorems, proofs,
  definitions, and algorithms where appropriate, logging every change in
  `reports/semantic_enrichment.json`.
- **Consolidated QA.** DeepSeek-V3 hallucination checks, Qwen2.5-7B log analysis,
  and the pdf2image diff gate now contribute to a single
  `reports/quality_gate.json` artifact that lists the models used and any
  flagged sections/pages. CI can fail fast by inspecting that file instead of
  parsing three different reports.

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
- `release/pipeline/reward_suite.py` — unifies heuristic + multimodal reward
  scoring (InternVL renders + Kimi-K2 judgments) and now exports
  `reward_trace.jsonl` for every run.
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
  - Default repo: `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct`; overridden by
    `LATEXIFY_LLM_REPO` or `--llm-repo`.
  - Falls back to `Qwen/Qwen2.5-1.5B-Instruct` (or custom entries) when GPU
    memory is tight via `LATEXIFY_LLM_FALLBACKS`.
  - vLLM helpers live in `release/models/vllm_client.py`; control the runtime
    checkpoint/VRAM budget via `LATEXIFY_VLLM_MODEL`,
    `LATEXIFY_VLLM_GPU_UTILIZATION`, and `LATEXIFY_VLLM_TP`. The bootstrapper now
    installs `llm/deepseek-v3` by default so the backend can start a local vLLM
    instance without hitting Hugging Face at runtime.
  - Style exemplars loaded from `release/reference_tex/<domain>/` using
    `core/reference_loader.py`.
  - Honours `HF_HOME`, `TRANSFORMERS_CACHE`, and `LATEXIFY_LLM_DIR` for weight
    storage.
- **OCR stack** (`release/pipeline/ingestion.py` + `release/models/model_adapters.py`)
  - Florence‑2 ➜ InternVL ➜ Nougat ➜ Tesseract run for every page when available.
  - Tune VRAM safety via `LATEXIFY_OCR_VRAM_HEADROOM_GB` (default 3 GiB) and override placement with `LATEXIFY_FORCE_GPU_OCR` (`1`, `cuda:1`, etc.) when you know the GPU has enough headroom.
  - `LATEXIFY_OCR_FORCE_GPU` lets you bypass the free-VRAM heuristic on a per-backend basis (`internvl`, `florence2`, `mathocr`, or `all`). Set `LATEXIFY_OCR_VRAM_HEADROOM_GB=0` + `LATEXIFY_OCR_FORCE_GPU=internvl` to keep InternVL on `cuda:0` even when Florence just ran.
  - Missing checkpoints auto-install (thanks to the bootstrapper) and the code
    clearly warns when optional wheels (`einops`, `timm`) are required.
  - Provenance for *all* OCR transcripts is stored on each chunk via
    `metadata["ocr_backends"]`.
  - A math-specific TrOCR backend (`--ocr-mode mathvision`) loads a
    VisionEncoderDecoder checkpoint fine-tuned on handwritten formulas for
    higher fidelity and automatically cascades after Nougat when running in
    `--ocr-mode auto`.
- **FlashAttention 2 (optional but recommended)**
  - Install inside `release/.venv` to trim several GB from Florence2 + InternVL attention layers:

    ```bash
    source release/.venv/bin/activate
    MAX_JOBS=8 CC=gcc-14 CXX=g++-14 NVCC_FLAGS="-allow-unsupported-compiler" \
      pip install --no-build-isolation --no-cache-dir flash-attn
    ```

  - The extra compiler flags satisfy CUDA’s host-compiler guardrails on Fedora 41+.
    If compilation still fails, install a GCC 14 toolchain (`dnf install gcc14 g++14`)
    and retry with the same command to avoid falling back to CPU/offload.
- **RAG cache** (`release/pipeline/rag.py`)
  - `rag.build_index(reference_tex/, cache_path)` parses curated `.tex` files into
    environment-level exemplars with type + domain tags.
  - Use `--rag-refresh` or `python release/scripts/rag_eval.py` to keep caches
    current.
- **Quality validators**
  - Section/headline hallucinations use DeepSeek-V3 via `hallucination.py`.
  - Compile log summaries run through Qwen2.5-7B (`LATEXIFY_VALIDATION_MODEL`).
  - Visual regression reports annotate differences with a Qwen2.5-VL style
    summary (`LATEXIFY_VISUAL_JUDGE_MODEL`).
  - Run `release/scripts/install_models.py --list` to see which checkpoints need
    `huggingface-cli login` before bootstrap; entries with `auth: huggingface-cli`
    in the listing require an authenticated session (DeepSeek, Mixtral, InternVL).
- **Multimodal Reward + Heuristic Scoring** (`release/pipeline/reward_suite.py`)
  - Merges syntax/semantic heuristics with InternVL renders and optional
    Kimi-K2 GGUF scores (via `release/models/kimi_k2_adapter.py`).
  - Requires Poppler + `pdf2image` so the first page can be rasterized and scored.
  - Override checkpoints with `LATEXIFY_MM_REWARD_MODEL_DIR` (InternVL) and
    `LATEXIFY_KIMI_K2_MODEL` / `LATEXIFY_KIMI_K2_VARIANT` if your GGUF files live
    outside `release/models/llm/kimi-k2-instruct-gguf/`.
- **Cross-validation layer.** `release/pipeline/cross_validation.py` compares branch coverage, content preservation, and layout fidelity before the reward is finalized. The scalar score feeds into `rewards.json["components"]["cross_validation"]`, `reward_trace.jsonl`, and the adaptive fusion router so snippet-level regressions surface even when aesthetic/semantic metrics look healthy.

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

- `python scripts/visual_regression_ci.py --run-dir build/runs/<run_id>` reruns the
  pdf2image diff gate in CI and fails the job when page drift is detected.
- `python scripts/active_learning_loop.py --runs-root build/runs --output training_data/active_learning/queue.jsonl`
  aggregates the per-run `active_learning_queue.jsonl` files into a single pending-review manifest.

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
  to back and collates metrics under `build/runs/benchmark_summary.json`.

## Outputs & Logs
- Every pipeline run writes into `build/runs/<run_id>/`.
- `artifacts/` holds ingestion caches (chunks, rasterizations, OCR scratch).
- `reports/` collects JSON payloads such as `plan.json`, `retrieval.json`, `snippets.json`, `metrics.json`, `rewards.json`, `agent_metrics.json`, and `reward_trace.jsonl`.
- Progressive enhancement artifacts:
  - `reports/input_quality.json` – preview/post-OCR quality profile.
  - `reports/domain_profile.json` – detected domain + package recommendations.
  - `reports/semantic_enrichment.json` – theorem/proof/algorithm upgrades.
  - `reports/quality_gate.json` – combined hallucination/validation/visual verdicts.
- `logs/` captures the streaming `checkpoint.log` used by the data-pathway logger.
- The root of the run directory still contains `main.tex`, `main.log`, optional `main.pdf`, and any other LaTeX artefacts from validation.

## Streaming Backend & UI
- `backend/app` exposes a lightweight FastAPI server that streams pipeline plans
  and stage status over Server Sent Events. The dispatcher now builds its demo
  plan via `release.pipeline.planner` so the UI reflects real chunk/section
  structure even when no PDF upload endpoint is wired yet.
- `apps/ui/gradio_app.py` consumes `/events` and renders the streamed plan +
  partial tokens; set `BACKEND=http://localhost:8000` (or your remote host) and
  run `uvicorn backend.app.main:app --reload` alongside the Gradio client for an
  end-to-end preview.
- The status bus exports the same stage names used in `DATA_PATHWAY.llm.jsonl`
  (`ingestion`, `planner`, `vision`, `synthesis`, `quality`), so dashboards and
  copilots can correlate UI updates with the run-directory telemetry.

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
- Heavy checkpoints can live outside the repo; set `LATEXIFY_MODELS_ROOT=/run/media/<user>/Active\ Storage/models`
  (or your preferred mount) and optionally `ln -s /run/media/<user>/Active\ Storage/models release/models` so
  installers and the pipeline read/write weights on that disk. All model download/install paths now honor the
  `LATEXIFY_MODELS_ROOT` override.
- Set `LATEXIFY_INTERNVL_MODEL=OpenGVLab/InternVL3_5-8B` (default) or another InternVL repo ID to right-size the
  OCR VLM for your GPU. The bootstrapper/installer automatically downloads to `models/ocr/<sanitized-id>`.
- Use `LATEXIFY_SEMANTIC_CHUNKER_ALLOW_DOWNLOAD=1` to let the semantic chunker
  pull `sentence-transformers/all-MiniLM-L6-v2` on demand, or pin the files in
  your offline cache ahead of time.
- Heavy OCR backends (Florence-2 ≈10 GB, InternVL ≈22 GB) reserve GPU memory in
  priority order (`LATEXIFY_OCR_GPU_PREF`, default `0` on the single 5090
  host). The runner auto-detects when only one ≤32 GB GPU is present and forces
  `LATEXIFY_OCR_RELEASE_MODE=page` so Florence and InternVL unload between
  pages instead of staying pinned. Override with
  `LATEXIFY_OCR_RELEASE_MODE=run` (and optionally
  `LATEXIFY_OCR_FORCE_HEAVY=1`) if you have spare VRAM and want them to remain
  resident; otherwise they raise when no GPU has enough headroom to keep the
  exact same models without silently falling back to heuristics.
- Sequential OCR is enabled by default (`LATEXIFY_OCR_SEQUENTIAL=1`): each heavy
  backend runs across the full document before the next one loads, so only one
  giant model occupies VRAM at a time. Set the variable to `0` if you prefer the
  earlier per-page cascade.
- `release/scripts/run_5090.sh` pins those defaults (sequential OCR, page-level
  unloads, heavy-only cascade) and routes LayoutLM/CLIP/LLM stages to CPU so a
  single 32 GB RTX 5090 never tries to host Florence and InternVL at the same
  time. Run `release/scripts/run_5090.sh release/inputs/<pdf>.pdf [extra args]`
  to apply the preset; it also records a lightweight `nvidia-smi` log under
  `release/logs/gpu_samples/` unless you set `RUN5090_DISABLE_GPU_LOG=1`.
- MathVision/TrOCR is CPU-friendly; enable it with `--ocr-mode mathvision` when
  transcribing formula-heavy scans or let the auto cascade invoke it between
  Nougat and Florence.
- PyMuPDF is preferred for PDF rasterization (and falls back to pdf2image if
  Poppler is installed). Installing dependencies via
  `python release/scripts/install_dependencies.py` brings PyMuPDF along.
- Every `run_release.py` invocation now performs a dependency preflight via
  `release/tools/dependency_installer.py`. It verifies critical Python
  packages (torch, transformers, PyMuPDF, psutil, llama-cpp, etc.) and
  system binaries (tectonic/latexmk/pdflatex, kpsewhich, Ghostscript,
  Poppler). Python packages are auto-installed into `release/.venv` when
  missing; system/LaTeX tools emit explicit distro-specific commands. Extend
  `release/tools/dependency_installer.py` whenever you introduce a new
  requirement so degraded fallbacks never sneak into release runs.
- The LLM refiner automatically falls back to CPU whenever no GPU exposes
  ≥6 GB free memory. Override with `LATEXIFY_LLM_DEVICE` or
  `LATEXIFY_LLM_MIN_FREE_GB` if you want a different threshold/placement.
- Fine-tuned LayoutLM weights can be loaded by pointing
  `LATEXIFY_LAYOUTLM_MODEL` at the directory produced by
  `scripts/train_layoutlm_token_classifier.py` (or any Hugging Face-compatible
  checkpoint). Combine this with `scripts/register_layout_dataset.py` +
  `training_runs/<slug>` to iterate quickly on document-structure models.
  LayoutLM now defaults to `auto`, so it lands on the first CUDA device when
  available; set `LATEXIFY_LAYOUTLM_DEVICE=cpu` if you explicitly want it off
  the GPU.
- On single-GPU hosts, heavy OCR backends (Florence2 / InternVL) load sequentially
  and unload after each page (`LATEXIFY_OCR_RELEASE_MODE=page` by default). Switch
  to `run` only if you have ample VRAM; otherwise the runner will fall back to CPU
  when the GPU cannot fit the second model.
- The optional CLIP verifier now defaults to `auto`, so it uses CUDA whenever
  available. Set `LATEXIFY_CLIP_DEVICE=cpu` if you need to keep the GPU free
  for OCR/LLM models.
- Install FlashAttention2 inside the release virtualenv
  (`pip install flash-attn --no-build-isolation`) to reduce Florence/InternVL
  VRAM usage; the adapters automatically switch to flash attention when present.
- On single-GPU hosts, set `LATEXIFY_PREFER_QWEN_VL=1` to swap the heavy
  InternVL backend for `Qwen2.5-VL-7B` by default. Tune Qwen-specific knobs via
  `LATEXIFY_QWEN_VL_MODEL`, `LATEXIFY_QWEN_VL_MAX_NEW_TOKENS`,
  `LATEXIFY_QWEN_VL_PROMPT`, and `LATEXIFY_QWEN_VL_TEMPERATURE` to balance VRAM
  usage (≈12–16 GB in FP16). Use `LATEXIFY_QWEN_VL_DEVICE=cpu`
  (or `cuda:0`/`auto`), `LATEXIFY_QWEN_VL_LOAD_IN_8BIT=1`, and
  `LATEXIFY_QWEN_VL_LOAD_IN_4BIT=1` to keep the checkpoint under tighter GPU
  budgets; the runner auto-falls back when OOMs exceed
  `LATEXIFY_QWEN_VL_MAX_GPU_RETRIES` (default 2).
- Set `LATEXIFY_DISABLE_VLLM=1` whenever you need every GB of VRAM for OCR;
  the runtime now honors this flag and skips instantiating the DeepSeek vLLM
  engine entirely, so InternVL can remain on CUDA.
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
- Attempt telemetry: `build/attempt_ledger.json` accumulates metadata for every
  `run_release.py` invocation (attempt id, config, runtime, status). Each
  `build/run-<run_id>/attempt_log.json` mirrors the latest record so debugging a
  single run is straightforward. The runner enforces a ~7200 s cumulative
  runtime cap; extend `release/tools/attempt_tracker.py` whenever you need to
  track new metrics.
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
