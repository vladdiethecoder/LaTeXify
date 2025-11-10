# LaTeXify · PROJECT_MAP

## Executive Overview
LaTeXify converts OCR/RAG bundles into clean, class-aware LaTeX, compiles PDFs, and self-heals TeX failures via a KB-guided loop.

**Subsystems**
- **Planning & Retrieval** — Build/query FAISS; assemble task bundles with MMR + deterministic fallbacks.
- **Synthesis & Routing** — Specialist router picks text/table/math/figure; llama.cpp GGUF backend for reproducible local gen.
- **Aggregation & Build** — Assemble snippets/assets, stage LiX classes, compile with `latexmk`, emit source maps.
- **Auto-Fix** — Parse logs, consult KB, patch macros/preamble/assets, recompile.

### C4-style Context (brief)
- **System**: Local CLI + scripts produce LaTeX/PDF from bundles and KB indices.
- **Containers**:
  - **Pipeline** (`latexify/pipeline/*`) — Retrieval, routing, planning, synthesis.
  - **KB Tools** (`latexify/kb/*`) — Build/query FAISS; alias dir; offline-friendly encoders.
  - **Agents** (`latexify/agents/*`) — CLIs for class probing/diagnostics.
  - **Build Tools** (`scripts/*`, `latexify/utils/*`) — Stage classes, aggregate, fix.
  - **Dev / Eval** (`dev/*`) — Prototypes, fixtures, QA utilities (**documented; excluded from JSONL**).

---

## Active Tree (indented by path)

main.py — Phase-2 entrypoint. [link](main.py)

latexify/
  __init__.py — Package marker. [link](latexify/__init__.py)

  agents/
    __init__.py — Marker. [link](latexify/agents/__init__.py)
    class_probe.py — **CLI**: compile probe; detect missing classes/fonts; JSONL log; fallback to `scrartcl`. [link](latexify/agents/class_probe.py)

  kb/
    __init__.py — Marker. [link](latexify/kb/__init__.py)
    build_latex_kb.py — Bootstrap minimal LaTeX KB JSONL; can spawn BGEM3 indexer. [link](latexify/kb/build_latex_kb.py)
    build_index_bgem3.py — BGE-M3 → FAISS (seeded; L2 + IP); `--prefix` naming. [link](latexify/kb/build_index_bgem3.py)
    build_index_compat.py — Chunks → FAISS (or `latex_docs.jsonl` fallback). [link](latexify/kb/build_index_compat.py)
    query_index.py — Load FAISS/meta; encode; over-fetch + MMR; write evidence JSON. [link](latexify/kb/query_index.py)
    ensure_kb_alias.py — Create **`kb/latex`** alias by linking/copying required files; can stage payload JSONLs. [link](latexify/kb/ensure_kb_alias.py)
    kb_paths.py — Required files + link-or-copy helpers. [link](latexify/kb/kb_paths.py)

kb/
  classes/ — Source LiX classes for staging to TEXMF (`.cls`, `lix.sty`). [link](kb/classes/tex/latex/lix)
  latex/ — **Alias target** consumed by retrieval; FAISS + metadata (`faiss.index`, `faiss.meta.json`, `latex_docs.*`, `chunks*`). [link](kb/latex)

latexify/pipeline/
  phase2_run_task.py — **CLI orchestrator**: retrieval → specialist → critic loop; writes snippet + operator trace. [link](latexify/pipeline/phase2_run_task.py)
  planner_scaffold.py — Normalize blocks/assets/consensus; add placeholders when visuals missing. [link](latexify/pipeline/planner_scaffold.py)
  retrieval_bundle.py — Build `ContextBundle`; multi-index search; evidence JSONL; graceful fallbacks. [link](latexify/pipeline/retrieval_bundle.py)
  specialist_router.py — Heuristic-first specialist selection; optional local GGUF classifier. [link](latexify/pipeline/specialist_router.py)
  model_router.py — Map task kind → model family. [link](latexify/pipeline/model_router.py)
  model_backends.py — llama.cpp backend wrapper (seeds, ctx, multi-GPU). [link](latexify/pipeline/model_backends.py)
  critic_agent.py — Deterministic critic; accept-first for tests. [link](latexify/pipeline/critic_agent.py)

  synth_shared.py — Sanitizers, slugify, capability detect, prompt cache. [link](latexify/pipeline/synth_shared.py)
  synth_text.py — Deterministic prose; emits TODOs when uncertain; returns capabilities. [link](latexify/pipeline/synth_text.py)
  synth_table.py — HTML/pipe-delimited → `booktabs` tabular. [link](latexify/pipeline/synth_table.py)
  synth_figure.py — `figure`/`table` env with safe captions/labels. [link](latexify/pipeline/synth_figure.py)
  synth_figure_placeholder.py — Placeholder for missing visuals; labeled with task id. [link](latexify/pipeline/synth_figure_placeholder.py)
  synth_formula.py — Labeled `equation`/`align`; returns `["amsmath"]`. [link](latexify/pipeline/synth_formula.py)
  **synth_latex.py** — CLI wrapper that loads bundles, routes specialists, and writes TeX + meta; deterministic when seeded. **[link](latexify/pipeline/synth_latex.py)**

latexify/utils/
  setup_classes.py — Stage LiX into project TEXMF; write `.latexmkrc`; idempotent. [link](latexify/utils/setup_classes.py)

scripts/
  aggregator.py — Assemble → `build/main.tex`; optional `latexmk`; source-map. [link](scripts/aggregator.py)
  aggregate_tex.py — Concatenate per plan. [link](scripts/aggregate_tex.py)
  auto_fix.py — Parse logs, fetch KB hints, patch, recompile; fix report. [link](scripts/auto_fix.py)
  approve_fixes.py — Promote successful auto-fixes into KB; optional reindex. [link](scripts/approve_fixes.py)
  block_consensus.py — Multi-backend consensus + disagreement flags. [link](scripts/block_consensus.py)
  build_chunks.py — Build role-aware chunks. [link](scripts/build_chunks.py)
  bin/class_selector.py — Decide class/capabilities from chunk stats. [link](scripts/bin/class_selector.py)
  **build_faiss_from_ocr.py** — Embed OCR JSONL and build FAISS for search; Sentence-Transformers backend. **[link](scripts/build_faiss_from_ocr.py)**

---

## Cross-Cutting
- **KB alias contract**: Consumers read **`kb/latex`** only. `ensure_kb_alias.py` guarantees `{faiss.index, faiss.meta.json, latex_docs.index, latex_docs.jsonl, latex_docs.meta.json}` + `chunks.jsonl/chunks/`.
- **Class staging**: `setup_classes.py` copies from **`kb/classes/tex/latex/lix`** into project TEXMF and writes `.latexmkrc` (`TEXMFHOME`/`TEXINPUTS`).
- **Config & Build**: See [pyproject.toml](pyproject.toml), [README.md](README.md), and optional [.env.local](.env.local) for local overrides.
- **Evidence**: Retrieval/orchestration write JSON/JSONL traces under `evidence/` and `dev/runs/*`.

---

## Inactive / Legacy / Duplicated (documented, **excluded from JSONL**)
- `latexify/ingestion/__init__.py` — Future ingest adapters (scaffold). [link](latexify/ingestion/__init__.py)
- `kb/offline/*` — Air-gapped payloads; not read by active pipeline. [link](kb/offline)
- `kb/online/github/LiX/*` — Mirrored upstream LiX repo; used to refresh `kb/classes`. [link](kb/online/github/LiX)
- `dev/*` — Prototypes, OCR backends, eval harness, fixtures, goldens, and tools. [link](dev)

---

## ADRs (embedded)
- **ADR-001: Keep JSONL index limited to active pipeline — Status: Accepted.**
- **ADR-003: KB alias via `kb/latex` — Status: Accepted.**
- **ADR-010: Project-local TEXMF via `.latexmkrc` — Status: Accepted.**
- **ADR-014: Keep class sources and alias separate — Status: Accepted.**

---

## Update Log
- **2025-11-06** — Added missing `synth_latex.py` and `scripts/build_faiss_from_ocr.py` to Active Tree; added Config & Build note; no changes to active/inactive classification.
