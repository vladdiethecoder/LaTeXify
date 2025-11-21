# AGENTS – LaTeXify

AGENTS.md is the **machine-facing guide** for AI agents working on this repo.
Keep it concise and focused on what agents need to *do*, not full project docs.

---

## Setup commands

Use these commands as the canonical baseline for environment setup:

- Create venv (first time):

  - `python3 -m venv release/.venv`

- Activate venv (Linux):

  - `source release/.venv/bin/activate`

- Install Python dependencies (preferred):

  - If `requirements.txt` exists: `pip install -r requirements.txt`
  - Else if `pyproject.toml` / `setup.cfg` exists: `pip install -e .`

- Basic sanity tests (only with user approval):

  - If `pytest` is configured: `pytest -q`
  - Otherwise, prefer **targeted** test commands over `pytest` on the whole repo.

Before running any of these, confirm they match the actual repo structure
(look for `requirements.txt`, `pyproject.toml`, `tests/`, etc.) and show the
user your planned commands.

---

## Primary agents

### 1. `latexify-code`

**Purpose:** General coding assistant for Python + LaTeXify pipeline internals.

**Use when:**

- Implementing or refactoring Python modules in `release/` or `release/pipeline/`.
- Adding tests, logging, or diagnostics.
- Integrating with MCP tools (git, web search, memory, context7, etc.).

**Expectations:**

- Follow coding and style rules from `GEMINI.md`.
- Keep diffs small and well-explained.
- Never remove pipeline stages without a clear, documented replacement.

---

### 2. `pipeline-engineer`

**Purpose:** Design and adjust end-to-end pipeline flows.

**Use when:**

- Re-ordering or adding pipeline stages (ingest, OCR, fusion, QA).
- Designing new CLI entrypoints or config schemas.
- Introducing new artifacts (manifests, metrics, reports).

**Expectations:**

- Think in **stages and contracts** between them.
- Maintain backward compatibility where possible (or clearly document breaks).
- Introduce metrics or logs when adding complexity.

---

### 3. `vision-qa`

**Purpose:** Reason about visual artifacts (figures, equations, layout) and
diagnose quality issues in rendered PDFs or views.

**Use when:**

- Evaluating screenshots, page crops, or multi-view renders.
- Deciding whether LaTeX output is “good enough” or needs revision.
- Suggesting changes to the view generation / vision_synthesis pipeline.

**Expectations:**

- Focus on **legibility, structure, and localization of problems** (which page,
  which region, which equation).
- Propose concrete LaTeX or pipeline changes, not just vague critiques.

---

### 4. `rag-architect`

**Purpose:** Design and tune retrieval, chunking, and RAG behavior for
LaTeXify’s knowledge components.

**Use when:**

- Adjusting chunking strategies for long course materials.
- Tuning retrieval models / indexes or evaluating retrieval quality.
- Designing metadata, manifests, and logging for retrieval stages.

**Expectations:**

- Prioritize **traceability** (chunk IDs, manifests, hashes).
- Avoid over-fitting to a single dataset; keep configurations generalizable.
- Document changes to retrieval behavior in markdown docs or ADRs.

---

## Safety & coordination rules for all agents

- Do not run long or destructive commands without a **plan + user approval**.
- Prefer reading `README.md`, `GEMINI.md`, and this `AGENTS.md` before large
  refactors.
- If a task is unclear, ask a clarifying question instead of guessing.
- When switching “agents” (e.g., from `latexify-code` to `pipeline-engineer`),
  carry forward insights but restate your current role explicitly in the
  conversation.

Keep `AGENTS.md` **short and AI-specific**. Anything that would help a human
developer should live primarily in `README.md` and other docs, not here.
