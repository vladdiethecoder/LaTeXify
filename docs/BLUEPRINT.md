# Analysis & Architectural Blueprint: PDF → LaTeX Conversion

This document captures a full post‑mortem of the current "LaTeXify" prototype and outlines a prescriptive architecture for a resilient, textbook‑quality PDF→LaTeX pipeline. It is intentionally opinionated so that future contributors share a single mental model of the desired end state.

## Part I • Post‑Mortem Findings

### 1.1 Strategic Misalignment – the Image Embedding Fallacy
- The prototype routinely emits `\includegraphics{...}` as the *primary* representation of text, math, and tables. The fallback is triggered whenever OCR/specialists struggle, silently replacing editable content with bitmaps.
- Consequences: no screen‑reader access, no searchability, pixelated zoom, zero editability, and a LaTeX source that is functionally identical to the original PDF. This violates the core product requirement (semantic, editable LaTeX).

### 1.2 Broken Document Understanding
- `master_plan.json` misclassifies heterogeneous regions as single block types (e.g., Q4(c) → `equation`), causing downstream agents to wrap plain text/tables inside math environments.
- Lack of document layout analysis (DLA) means the pipeline never distinguishes between key-value tables, instruction paragraphs, and math derivations. Specialists therefore cannot apply the right template.

### 1.3 Predictable Compilation Failures
- Sample errors from `main.log` include `Misplaced alignment tab character &` and `Undefined control sequence \inR`. Root causes:
  - Blocks labeled as `equation` include literal strings like "Domain & Range", so `&` is treated as an alignment marker inside math mode.
  - Non-standard macros (e.g., `\inR`) are emitted instead of canonical `\in \mathbb{R}` because OCR text was never normalized.
- These errors are deterministic symptoms of the misclassified plan and the "wrap everything in math" generation path.

## Part II • Target Output Standard (“Textbook” Baseline)

### 2.1 Document Class & Core Packages
- Use a book-like class: `\documentclass[12pt]{book}` or `memoir`. Ditch `article`/`twocolumn`.
- Baseline packages for semantic fidelity:
  - `geometry`, `graphicx`, `amsmath`, `amsthm`, `amsfonts`, `amssymb`
  - `booktabs`, `siunitx` for tables & numerical fidelity
  - `tcolorbox` + `enumitem` for styled question/answer environments

### 2.2 Semantic Blocks (tcolorbox)
- Define a reusable `question` environment via `tcolorbox` (auto-numbered, breakable, shaded) to house each problem statement + solution.
- Pair with `answer` or context-specific boxes (definitions, examples) so the LaTeX intentionally mirrors textbook conventions.

### 2.3 Professional Tables
- Use `booktabs` (\toprule / \midrule / \bottomrule) for structures like the Q4 summary cards.
- Escape textual ampersands (`\&`) and keep math inline using `$…$` or `\(…\)`. Only genuine equations should reach math environments.

### 2.4 Multi-part Problems & Alignments
- Model multi-step algebra as `enumerate` + `align*` blocks. This enforces alignment on `=` and produces readable derivations.

### 2.5 Figures
- Reserve `\includegraphics` for true figures. Use `figure` + paired `minipage` layouts for side-by-side assets. All explanatory text stays in LaTeX paragraphs, not captions.

## Part III • Architectural Blueprint for a Robust Pipeline

### 3.1 Phase 1 – Multimodal Document Layout Analysis (DLA)
- **Goal:** produce a semantic master plan before any OCR/text generation.
- **Implementation:** Vision–language model (e.g., Florence2, DocFormer, LayoutLMv3, Pix2Struct) processes page images + structural cues to classify regions (`section-header`, `key-value-table`, `figure`, `formula`, `solution-text`).
- **Output:** JSON plan with bounding boxes, logical roles, and `content-ref` IDs. This acts as the contract for downstream specialists.

### 3.2 Phase 2 – Specialist Agents per Modality
- **Math OCR Agent:** takes `formula` / `solution-algebra` crops, runs Nougat/Mathpix/TrOCR to emit TeX tokens.
- **Table Agent:** uses Table-Transformer or a layout-aware model to emit structured rows/columns or key/value pairs.
- **Text Agent:** handles prose via OCR+LLM rewriting, preserving semantic line breaks and applying domain heuristics.
- **Figure Agent:** de-duplicates and crops images, ensuring captions are textual.
- **Classification Agent:** labels instructions vs. solution text vs. metadata to drive templates.

### 3.3 Phase 3 – Template-Based Generation
- Map each semantic role to a vetted LaTeX template:
  - `section-header` → `\chapter`/`\section`
  - `problem-instruction` + `solution-algebra` → `\begin{question} ... \begin{align*} ...`
  - `key-value-table` → `booktabs` table with `\caption`
  - `figure-pair` → `figure` + `minipage`
- This approach eliminates “raw string” LaTeX generation and keeps outputs deterministic.

### 3.4 Phase 4 – Self-Healing Validation Loop
1. **Generate:** Assemble `main.tex` from templates + snippets.
2. **Compile:** Run `latexmk -pdf -interaction=nonstopmode main.tex` in an isolated build dir.
3. **Parse Logs:** Use a parser (e.g., `texoutparse`) to extract errors/warnings/bad boxes into structured diagnostics.
4. **Repair:** Feed the failing snippet + original semantic chunk into a "debug" agent to emit a patch (escape `&`, replace `\inR`, etc.).
5. **Repeat:** Iterate until the document compiles cleanly or a hard stop threshold is reached.

This CI-like loop ensures regressions never ship silently and gives the pipeline a way to learn from its mistakes.

## Action Items

1. **Implement DLA:** Integrate PyMuPDF/Florence2 features with the new semantic chunker to emit `LayoutRegion` metadata for every chunk (already started in `release/pipeline/ingestion.py`).
2. **Template Library:** Codify `question`, `answer`, `table`, and `figure` templates and ensure specialists only fill the payloads.
3. **Math/Table OCR:** Wire MathVision/Nougat outputs and structured table detectors into the OCR fallback chain.
4. **Validation Harness:** Build the log parser + repair agent scaffolding so `run_release.py` fails fast and can auto-fix trivial issues.
5. **Documentation & QA:** Keep this blueprint updated as the architecture evolves and add automated checks (CI) for compilation success plus schema validation for plans/events.

Delivering the "academic textbook" standard requires treating LaTeX generation as an engineered system, not a monolithic LLM prompt. The blueprint above translates the lessons from the current failure into a concrete, testable architecture.
