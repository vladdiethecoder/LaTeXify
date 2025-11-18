# LaTeXify Mathematical Document Processing Plan

This document distills the 15-point improvement plan for upgrading LaTeXify into a fully local, high-quality mathematical document pipeline. Each section lists the goal, implementation sketch, and required open-source tooling.

## 1. Enhanced Mathematical OCR Pipeline
- **Goal:** Replace generic OCR with math-aware detectors + recognizers.
- **Implementation:** Train a YOLOv8 variant for math region detection and pair it with the `pix2tex` stack (or a custom LaTeXifyOCR wrapper). Add a syntax validator that enforces balanced delimiters and legal commands.
- **Deliverables:** `MathOCREngine` class with ensemble voting across pix2tex, a Mathpix-compatible clone, and symbol-by-symbol parsing.

## 2. Structured Document Understanding
- **Goal:** Understand page layout before generation.
- **Implementation:** Fine-tune `LayoutLMv3-base` (or DocFormer) on worksheets to classify headers, question stems, work areas, tables, etc. Expose a `DocumentStructureAnalyzer` that produces a document tree per page.

## 3. Mathematical Content Classification
- **Goal:** Tag each text chunk with its mathematical role.
- **Implementation:** Combine regex/rule patterns with ML classifiers to distinguish inline math, display equations, derivations, proofs, and function definitions. Output typed blocks for downstream templates.

## 4. Template-Driven LaTeX Generation
- **Goal:** Generate deterministic, vetted LaTeX via Jinja2 templates.
- **Implementation:** Maintain templates for `math_worksheet`, `proof_document`, and `problem_set`. The `LaTeXTemplatingEngine` loads the correct template and renders it using structured content blocks.

## 5. Local LLM Orchestration with Constrained Generation
- **Goal:** Keep all LLM inference local and grammar-constrained.
- **Implementation:** Run `codellama-13b-math` (quantized via llama.cpp) with a math-specific grammar file so inline/display math, `align`, `cases`, etc., follow strict syntax.

## 6. Mathematical Consistency Validation
- **Goal:** Ensure generated math preserves variables/operators.
- **Implementation:** Implement symbolic checks (via SymPy/latex2sympy) for variable sets, operator presence, structural shape, and semantic equivalence when feasible.

## 7. Multi-Model Ensembles
- **Goal:** Increase robustness with consensus.
- **Implementation:** For each critical task (OCR, structure detection, generation), run multiple models (e.g., pix2tex, DonutMath, CodeLlamaMath) and weight their outputs before committing to a result.

## 8. Progressive Document Assembly
- **Goal:** Build `main.tex` section-by-section with validation gates.
- **Implementation:** `ProgressiveAssembler` generates each section, runs validation, and falls back to conservative formatting if a block fails.

## 9. Math Environment Detection
- **Goal:** Choose the correct LaTeX environment automatically.
- **Implementation:** Pattern detectors decide between `align`, `gather`, `cases`, `matrix`, etc., based on alignment characters, line breaks, piecewise cues, or matrix-like token sequences.

## 10. Error Recovery & Repair
- **Goal:** Automatically fix common LaTeX issues.
- **Implementation:** `LaTeXErrorRepair` parses compilation logs (via texoutparse) and applies targeted fixes for missing delimiters, undefined commands, alignment errors, and overfull boxes.

## 11. Content-Aware Chunking
- **Goal:** Chunk by semantic boundaries rather than character counts.
- **Implementation:** The `SemanticChunker` inspects headers, numbering, and math environments to decide chunk boundaries (Question N, \section, \begin{equation}, list items, proofs, etc.).

## 12. Symbol Normalization
- **Goal:** Ensure consistent Unicode→LaTeX conversion.
- **Implementation:** Maintain a comprehensive normalization map for arithmetic symbols, relations, arrows, and Greek letters, applied before generation.

## 13. Quality Assessment Pipeline
- **Goal:** Score outputs beyond simple compilation checks.
- **Implementation:** Metrics include math preservation (symbolic similarity), structural fidelity (layout comparison), syntactic correctness (compile success), semantic accuracy (LLM cross-check), and aesthetics (typographic heuristics).

## 14. Iterative Refinement Loop
- **Goal:** Re-run weak sections until the document meets thresholds.
- **Implementation:** `IterativeRefiner` uses the quality assessor to locate poor sections, regenerates them with improved prompts/context, and re-assembles the document up to N iterations.

## 15. Domain-Specific Prompt Engineering
- **Goal:** Provide specialized prompts per content type.
- **Implementation:** Maintain prompt templates for equation conversion, proof formatting, table reconstruction, etc., so the LLM operates under explicit math rules.

## Roadmap
### Phase 1 (Weeks 1–4)
1. Stand up local LLM inference (CodeLlama Math via llama.cpp).
2. Integrate pix2tex and wrap it inside `MathOCREngine`.
3. Implement the Jinja2 template system.
4. Build the error detection + repair scaffold.

### Phase 2 (Weeks 5–8)
5. Fine-tune `LayoutLMv3` for math documents.
6. Implement ensemble processors.
7. Ship the quality assessment suite.
8. Add math consistency validators.

### Phase 3 (Weeks 9–12)
9. Wire up iterative refinement.
10. Add progressive assembly logic.
11. Deploy the semantic chunker.
12. Enhance prompt engineering / symbol normalization.

### Phase 4 (Weeks 13–16)
13. Run comprehensive dataset tests (handwritten + printed math sets).
14. Optimize for latency/memory (quantized models, batching).
15. Finalize documentation + deployment scripts.

## Resource Requirements
- **Compute:** Single RTX 5090 (32 GB VRAM) GPU, 32 GB RAM, ~500 GB disk for models/datasets.
- **Models:** `codellama-math`, `pix2tex`, `LayoutLMv3`, YOLOv8 math detector.
- **Dependencies:** `torch`, `transformers`, `ultralytics`, `pix2tex`, `sympy`, `latex2sympy`, `pdfplumber`, `PyMuPDF`, `llama-cpp-python`, `vllm`, `huggingface_hub`.

This plan keeps the toolchain entirely local/free while dramatically improving mathematical fidelity, structural accuracy, and robustness.
