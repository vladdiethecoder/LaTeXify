# ADR: Local PDF → LaTeX Stack

- **Date:** 2025-11-18
- **Status:** Accepted

## Decision / Status
- **Target:** Local, open-source PDF→LaTeX ingestion and orchestration stack owned inside LaTeXify.
- **Constraints:** ≤32 GB GPU VRAM footprint, Python-first tooling, strong math/table support, offline-only (no SaaS dependencies).

## Context

This Architecture Decision Record captures the latest research pass on locally runnable, open-source components that could lift LaTeXify’s ingestion, OCR, and orchestration fidelity. The goal is to anchor the research note in the canonical docs set so future changes can reference a durable source of truth.

## External Landscape Scan — PDF → LaTeX Stack (Nov 2025)

### Search Metadata
- **Goal**: Discover models, tools, and libraries that improve a local, open-source PDF → LaTeX pipeline, especially layout analysis, math OCR, and RAG orchestration.
- **Scope**: models, tools/libraries, RAG components, and end-to-end architectures.
- **Constraints**: local-only execution, open-source licensing, ≤32 GB GPU VRAM footprint, Python-friendly tooling.
- **Nice-to-have**: strong math + table support, active maintenance, and straightforward Hugging Face integration.

### Candidate Findings
#### MinerU (architecture) — fit score 0.95
- **Why it matters**: 1.2 B param multimodal stack purpose-built for document understanding; handles layout, formulas, and tables in a unified two-stage flow (layout → content).
- **Benefits to LaTeXify**: aligns with the current “analyze layout then transcribe content” blueprint, reaches SOTA accuracy on complex PDFs, and stays within a single high-memory GPU, reducing fallback churn.
- **Integration notes**: Could replace or augment the Florence/InternVL ensemble. Emits structured content blocks that can flow directly into chunking/planning.
- **Status**: Very active (latest 2.6.4 release in Nov-2025). License listed as open-source on GitHub (confirm exact terms before shipping).

#### LaTeX-OCR / pix2tex (model) — fit score 0.90
- **Why it matters**: Specialized ViT+ResNet encoder-decoder that converts equation crops into LaTeX (printed + handwritten).
- **Benefits to LaTeXify**: Inserts as a high-accuracy math OCR specialist, reducing bitmap fallbacks and easing downstream normalization.
- **Integration notes**: Already familiar tech (pix2tex). Can be slotted into `mathocr` refinement or post-OCR repair loops for any figure-equation crops.
- **Status**: Open-source, locally runnable, community-maintained; confirm latest checkpoints & license variant used.

#### PDF Processor MCP Server (tool/library) — fit score 0.85
- **Why it matters**: Reference MCP server wiring PyMuPDF + pix2tex into a deterministic pipeline targeted at academic PDFs.
- **Benefits to LaTeXify**: Useful blueprint for combining rule-based text extraction with math-specific OCR in a modular service; reinforces the “local, inspectable” requirement.
- **Integration notes**: Even if MCP is not adopted, its codebase can guide improvements to `ocr_helper` orchestration or become a drop-in service reachable via MCP adapters.
- **Status**: Open-source, released June 2025, currently discussed in recent community write-ups (license needs verification).

#### Marker (tool/library) — fit score 0.85
- **Why it matters**: End-to-end PDF-to-text/LaTeX converter optimized for math-heavy documents and reportedly outperforming MathPix.
- **Benefits to LaTeXify**: Provides both a benchmark and a potential ingestion fallback; strong math handling aligns with our highest-risk failure mode.
- **Integration notes**: Evaluate as a competing ingestion backend (maybe `marker` mode) or mine its heuristics for equation/table stitching.
- **Status**: Actively developed (blog post Sept-2025). Verify license before redistribution.

#### LangGraph (RAG/orchestration) — fit score 0.80
- **Why it matters**: Stateful LangChain extension for multi-actor workflows with loops/persistence.
- **Benefits to LaTeXify**: Could coordinate compile/repair, specialist dispatch, and ambiguity resolution as a graph instead of bespoke pipelines.
- **Integration notes**: Candidate abstraction for the “planning/assembly” layer that handles repair cycles and stateful decisions.
- **Status**: MIT-licensed, actively maintained through 2024–2025.

#### LlamaIndex (RAG component) — fit score 0.80
- **Why it matters**: Mature data framework for ingesting, indexing, and querying corpora—ideal for “reference TeX” retrieval.
- **Benefits to LaTeXify**: Powers semantic search over curated LaTeX exemplars so agents can ground outputs in high-quality patterns.
- **Integration notes**: Drop into the RAG stage after chunking; pair with LangGraph (or existing planners) to retrieve relevant snippets before synthesis.
- **Status**: MIT-licensed, actively maintained through 2024–2025.

### Prioritized Short List
- **Top picks for near-term evaluation**: MinerU, LaTeX-OCR (pix2tex), PDF Processor MCP Server, Marker.
- **Watchlist / mid-term**: LangGraph, LlamaIndex (investigate when orchestration or RAG upgrades rise in priority).

### Outstanding Questions
1. Do downstream consumers require specific permissive licenses (MIT/Apache) for redistribution?
2. Which pipeline stage is the most urgent bottleneck (initial ingestion, layout, math OCR, or RAG/planning)?
3. Are we targeting wholesale replacement of the DLA stack or incremental bolt-ons around Florence/InternVL?
