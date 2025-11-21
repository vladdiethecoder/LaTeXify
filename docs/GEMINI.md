# GEMINI – LaTeXify Project Guide

You are collaborating on the **LaTeXify** project: an AI-driven pipeline that
turns messy course materials (handwritten notes, typed PDFs, slides, etc.) into
high-quality, LiX-class LaTeX textbooks with automated QA.

This file gives you **project-specific rules and context** for this repository.
System-level behavior, meta-prompting, and tool orchestration live in
`.gemini/system.md`.

---

## Project goals

- Convert heterogeneous educational inputs into clean, compilable LaTeX.
- Maintain a **reproducible pipeline** with clear stages (ingest → OCR →
  structure → LaTeX synthesis → vision QA → auto-fix → final PDF).
- Prioritize **correctness, debuggability, and traceability** over raw speed.
- Keep the repo safe for heavy GPU workloads (no reckless commands).

---

## Tech stack & environment

- **Language:** Python 3.11+.
- **Core libs:** PyTorch + CUDA, scientific Python stack, LaTeX toolchain.
- **Pipeline code:** under `release/` and `release/pipeline/` (e.g.
  `vision_synthesis.py`, `vision_agents.py`, `snippet_fusion.py`, etc.).
- **Entry points:** top-level orchestrators such as `run_release.py` and
  small helper scripts under `release/` and `scripts/` (if present).
- **Tools & integrations:** MCP servers for Git, web search, context7, memory,
  core_memory, GitHub, Playwright, Chrome DevTools, and sequential thinking.

When unsure about a command or module, **inspect the code and README first**.
Never invent critical commands from thin air if you can discover them in the
repo.

---

## Repository layout (high-level expectations)

Treat these as *guiding expectations*:

- `release/` – production pipeline code, configs, and orchestrators.
- `release/pipeline/` – modular stages (chunking, OCR, vision synthesis,
  snippet fusion, QA agents).
- `tests/` or `release/tests/` – unit/integration tests (if present).
- `docs/` – design docs, ADRs, and pipeline diagrams (if present).
- `data/` or `examples/` – sample inputs/outputs (never rewrite originals).

Before large edits, use shell / file tooling (ls, tree, ripgrep) to confirm
actual layout and adjust your plan accordingly.

---

## How to work with this repo

### 1. Always start with a plan

For any non-trivial request:

1. Restate the user’s goal in 1-3 sentences.
2. Propose a **numbered plan** of steps (read files → reason → edits →
   tests/commands → expected artifacts).
3. Wait for approval or adjust based on feedback before editing many files.

Plans should reference **specific files and functions**, not vague “update the
pipeline”.

### 2. Editing rules

- Prefer **small, focused diffs** over broad rewrites.
- Preserve comments and TODOs unless they’re clearly wrong or obsolete.
- Never auto-reformat the whole file or repo unless explicitly asked.
- For pipelines, favor **adding small composable functions** over adding more
flags to giant functions.

### 3. Quality & tests

- Before adding complex features, **look for existing tests** and patterns.
- When you touch logic, prefer to:
  - Add or update tests alongside it.
  - Suggest new tests even if you are not asked explicitly.
- If the user has not specified the exact test command, assume a conservative
workflow:
  - Propose a test command (e.g. `pytest` or `python -m pytest`) **in the plan**
    and wait for confirmation before running it.
- Treat any failing test as a **signal to investigate**, not a reason to
comment it out.

---

## Style guidelines

### Python

- Use type hints for new code; keep signatures explicit.
- Prefer simple, flat functions over deep class hierarchies.
- Keep I/O (filesystem, network, GPU) **separated** from pure logic where
possible to make testing easier.
- Log important events at key pipeline stages (inputs, config, metrics,
failures), but avoid noisy logging inside tight inner loops.

### LaTeX

- Generate **minimal, clean LaTeX**:
  - Avoid unnecessary packages; reuse project-standard preambles if present.
  - Prefer semantic structure (sections, equations, environments) over manual
    spacing hacks.
- Never introduce non-portable LaTeX that requires obscure custom engines unless
the repo already uses them.

---

## Things you must NOT do

- Do **not**:
  - Delete or overwrite raw input data under `data/` / `examples/`.
  - Commit or generate large binary artifacts into source folders unless that is
    the established pattern.
  - Run extremely heavy GPU jobs or multi-hour scripts without user approval.
  - Circumvent missing dependencies by silently skipping steps; instead, follow
    the strategy in `.gemini/system.md` (diagnose, generalize installers, log
    findings).

When in doubt, **ask for clarification** or propose options rather than guessing.

---

## Where to look for deeper context

When you need more detail:

- Start with `README.md` / `docs/` / `AGENTS.md`.
- For pipeline specifics, read `release/README.md`, `run_release.py`, and
  `release/pipeline/*.py`.
- For tool behavior, inspect `.gemini/settings.json` and any MCP-related docs.

Keep this file as the **high-level project manual**; do not duplicate full
design docs here—link or reference them instead.
