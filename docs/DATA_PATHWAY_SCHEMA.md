# Data Pathway Schema

## Overview
Every LaTeXify run now emits a **data pathway log** so humans and LLM copilots can
reconstruct how `main.tex` / `main.pdf` were produced. The log lives next to the
build artifacts for that run and captures both a machine-friendly JSONL stream
and a short Markdown narrative.

```
build/
└── run-<run_id>/
    ├── DATA_PATHWAY.llm.jsonl
    └── DATA_PATHWAY.md
```

- `DATA_PATHWAY.llm.jsonl` – append-only, one event per line, optimized for fast parsing.
- `DATA_PATHWAY.md` – concise human summary with links to the key artifacts.

Together they describe the data lineage from the input PDF, through OCR/layout,
into LaTeX/PDF outputs.

## JSONL Event Schema
Each line is a compact JSON object with the following keys:

| Field          | Type              | Description                                                                 | Example                                    |
|----------------|-------------------|-----------------------------------------------------------------------------|--------------------------------------------|
| `timestamp_utc`| string            | ISO 8601 timestamp (UTC) when the event was recorded.                       | `"2025-11-17T02:55:24.112Z"`                |
| `run_id`       | string            | Stable identifier shared by all events for the run (`sample_20251116_2349`).| `"sample_20251116_234919"`                 |
| `stage`        | string            | Logical stage name (`ingestion`, `synthesis`, `reward`, ...).               | `"assembly"`                               |
| `status`       | string            | `"started"`, `"completed"`, or `"failed"`.                                  | `"completed"`                              |
| `input_files`  | list of strings   | Relative paths to primary inputs consumed during the stage.                 | `["release/samples/sample.pdf"]`           |
| `output_files` | list of strings   | Relative paths to important outputs created during the stage.               | `["build/runs/sample/main.tex"]`      |
| `models`       | list of strings   | Model identifiers used by the stage (LLMs, OCR models, etc.).              | `["Qwen/Qwen2.5-1.5B-Instruct"]`           |
| `metrics`      | object            | Optional structured metrics emitted by the stage (vision coverage, etc.).  | `{"vision_branch":{"coverage":0.62}}` |
| `metadata`     | object            | Optional provenance data (branch info, orchestrator notes, etc.).          | `{ "branch": { "name": "branch_b" } }` |
| `notes`        | string            | Short free-text context (max ~200 chars).                                   | `"Assembly completed (compile enabled)."`  |

Fields are always present (empty lists when unused) so downstream tools can rely
on a stable schema. Events are recorded sequentially, enabling timeline replay.
The optional `metrics` object captures structured telemetry (e.g., vision branch
coverage, snippet fusion confidence) so agents can correlate stage outcomes
without parsing human-readable notes.

### Metric Keys
Common metric payloads now include:

- `layout_confidence` – reported by `ingestion`; includes `avg`, `min`, and the
  configured threshold so agents can judge how trustworthy the layout metadata is.
- `cross_validation` – emitted after reward scoring and contains the latest
  structural/content/semantic composites as well as the run-wide confidence.
- `robust_compilation` – records `compilation_attempts`, full attempt history,
  and `recovery_success` flags so monitors know whether the fallback logic was
  engaged.
- `kimi_k2` – tracks `calls`, `avg_time_ms`, `success_rate`, and
  `repair_success_rate` for semantic validation / repair runs.

Additional per-stage metrics (visual regression, snippet fusion, etc.) may be
added over time; consumers should treat unknown keys as optional extensions.

### Branch Events
Branch sequencing emits two additional stage types:

- `parallel_branches` records the outcome of Branch A/B/C execution. Its
  metrics capture the aggregate completion counts, while the metadata payload
  references the manifest at `reports/branch_artifacts.json` so tools can trace
  which branch produced which intermediate files.
- `branch_outputs` fires after the best branch is selected. The stage links to
  `reports/branch_outputs.json` and lists the per-branch sample assets
  (`branch_a/sample_a.pdf`, etc.). The legacy `main.tex` / `main.pdf` in the run
  directory always mirror the branch chosen here, preserving backward
  compatibility for downstream consumers.
- `branch_evaluation` captures the BLEU-style comparison + visual fidelity
  report produced by `release/pipeline/branch_evaluator.py`. This stage references
  `reports/branch_evaluation.json` and exposes metrics such as
  `avg_bleu`, `compile_success_rate`, and the list of low-confidence chunks so
  active-learning agents and dashboards can contrast branch performance without
  reprocessing artifacts.
- `constraint_maps` is emitted when `--emit-constraint-maps` or
  `--enable-render-aware` is active. It links to
  `reports/constraint_maps.json`, lists the constraint/mask PNGs, and exposes
  metrics for `pages` and `rendered_items`.
- `render_aware` fires after Flux (or other diffusion backends) synthesize
  photorealistic pages. It references `reports/render_aware.json`, logs the
  rendered PNGs, and emits metrics for `pages`/`renders` so dashboards can
  compare render-aware runs without opening the artifact directory.

### Backend / SSE status events
The FastAPI backend (`backend/app`) forwards the same `stage` names over its
Server Sent Events feed (via `StageForwarder`). UI clients can therefore map
`status` updates from `/events` directly to `DATA_PATHWAY.llm.jsonl` entries and
link streamed tokens back to the corresponding planner blocks.

### Example JSONL Snippet
```jsonl
{"timestamp_utc":"2025-11-17T02:55:24.112Z","run_id":"sample_20251116_234919","stage":"ingestion","status":"started","input_files":["release/samples/sample.pdf"],"output_files":[],"models":["nougat-small","florence2-large","internvl:OpenGVLab/InternVL3_5-8B","trocr-math","pix2tex-base"],"notes":"chunk_chars=1200"}
{"timestamp_utc":"2025-11-17T02:56:31.404Z","run_id":"sample_20251116_234919","stage":"assembly","status":"completed","input_files":["build/runs/sample_20251116_234919/reports/plan.json","build/runs/sample_20251116_234919/reports/snippets.json"],"output_files":["build/runs/sample_20251116_234919/main.tex","build/runs/sample_20251116_234919/main.pdf"],"models":[],"notes":"Building LaTeX and assets."}
{"timestamp_utc":"2025-11-17T02:56:48.007Z","run_id":"sample_20251116_234919","stage":"run","status":"completed","input_files":["release/samples/sample.pdf"],"output_files":["build/runs/sample_20251116_234919/main.tex","build/runs/sample_20251116_234919/main.pdf"],"models":[],"notes":"duration=83.72s"}
```

## DATA_PATHWAY.md
The Markdown companion provides:
- Title line (`# Data Pathway — Run <run_id>`).
- Quick facts (status, input PDF, output paths, run duration).
- Optional configuration snapshot (chunk length, reward mode, etc.).
- Bullet list summarizing each stage and its outcome.

Both files are deterministic and token-efficient, enabling an LLM to stream the
JSON lines for structured reasoning while a human can skim the Markdown summary.

## Usage Notes
1. Locate the relevant build directory. Each pipeline invocation logs to
   `build/run-<run_id>/`. The `<run_id>` matches the artifact folder under
   `build/runs/<run_id>/`.
2. Open `DATA_PATHWAY.md` for a quick overview (input PDF, output files, status).
3. Stream or parse `DATA_PATHWAY.llm.jsonl` to reconstruct the timeline:
   - Search for `stage="ingestion"` to find OCR details.
   - Inspect `stage="assembly"` to see where `main.tex` was written.
   - Confirm `stage="run"` has `status="completed"` before trusting artifacts.

External agents (ChatGPT, Gemini, etc.) should read `docs/PROJECT_OVERVIEW.md`
first to understand the high-level system, then use the per-run logs to
reconstruct how any `main.tex` / `main.pdf` was produced.
