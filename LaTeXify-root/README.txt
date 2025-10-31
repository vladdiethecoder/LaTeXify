# LaTeXify

## Project Overview

LaTeXify is an advanced document processing pipeline designed to convert input documents (such as PDFs) into high-quality, compilable LaTeX code. A multi-agent, Retrieval-Augmented Generation (RAG) pipeline to convert PDFs into compilable LaTeX. The system uses advanced layout analysis and a "prompt-as-code" framework to deconstruct, plan, and synthesize documents. It leverages a combination of Optical Character Recognition (OCR), Retrieval-Augmented Generation (RAG), and intelligent LaTeX-aware agents to manage the complex task of document synthesis.

The system is built around a multi-stage pipeline that includes:
1.  **Ingestion & OCR:** Processes input files (e.g., PDFs) and extracts text and layout information using an ensemble of OCR backends.
2.  **Chunking & Indexing:** Intelligently chunks the extracted content and builds a searchable vector index.
3.  **Document Planning:** Analyzes the document content to select an appropriate LaTeX document class (e.g., article, textbook, thesis) and plans the overall structure.
4.  **Knowledge Retrieval:** Queries a dedicated LaTeX knowledge base (KB) to retrieve necessary packages, commands, and environment templates.
5.  **LaTeX Synthesis:** Generates the final `.tex` file snippet by snippet, assembling the preamble, title, body content, and document-ending commands.
6.  **Compilation & Post-processing:** Compiles the generated LaTeX code and performs any necessary fix-ups to ensure a valid PDF output.

## Core Architecture

The conversion process is a modular, multi-stage pipeline:

1.  **Orchestration:** `scripts/pipeline.py` manages the end-to-end flow.
2.  **Layout Analysis:** The `/pdf-document-layout-analysis/` submodule (using VGT/DiT models) and `scripts/ocr_ensemble.py` extract structured data (text, tables, figures) from the PDF.
3.  **Agents & Prompts:** Core logic is driven by agents in `/scripts/` using prompts from `/tasks/`.
    * `choose_doc_class.py`: Selects the best LaTeX class (e.g., 'article', 'textbook') from `/kb/classes/`.
    * `planner_scaffold.py`: Creates a structural `plan.json` for the document (e.g., [TITLE, Q1, Q2, ...]).
    * `retrieval_agent.py`: Queries the LaTeX KB (`/kb/latex/`) for each item in the plan.
    * `synth_latex.py`: Generates a `.tex` snippet for each plan item using its specific data bundle.
    * `aggregator.py`: Assembles all snippets from `/build/snippets/` into the final `build/main.tex`.
4.  **Data Flow:**
    * `/kb/`: Source of truth for RAG (LaTeX commands, class profiles).
    * `/bundles/`: Data packets (OCR + RAG context) for the Synthesis agent.
    * `/build/`: Output directory for generated snippets, `plan.json`, and the final `main.tex`.
5.  **Evaluation:**
    * `/dev/runs/`: Contains end-to-end test runs on various document types (e.g., `assignment_e2e`).
    * `/dev/eval/metrics.py`: Script for scoring output quality.
    * `/tests/`: Pytest suite for core components.

## Key Components

-   `/scripts/`: Core Python scripts driving the pipeline.
    -   `pipeline.py`: Main orchestration script.
    -   `ingest_pdf.py`, `ocr_ensemble.py`: Handles file ingestion and OCR.
    -   `build_chunks.py`, `build_index.py`: Manages content chunking and vector store creation.
    -   `choose_doc_class.py`: Selects the LaTeX document class.
    -   `retrieval_agent.py`, `query_index.py`: Manages retrieval from the KBs.
    -   `synth_latex.py`: The final LaTeX generation agent.
-   `/kb/`: Knowledge Bases
    -   `/latex/`: A vector-indexed KB for general LaTeX commands, packages, and best practices.
    -   `/classes/`: Profiles and templates for different LaTeX document classes.
-   `/dev/`: Development scripts, experimental backends, and test runs.
-   `/tasks/`: Markdown files defining the prompts and goals for different agents in the pipeline (e.g., planner, synthesizer).
-   `/build/`: Output directory for generated `.tex` files, logs, and other artifacts.
