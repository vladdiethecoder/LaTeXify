# LaTeXify

## Project Overview

LaTeXify is an advanced document processing pipeline designed to convert input documents (such as PDFs) into high-quality, compilable LaTeX code. It leverages a combination of Optical Character Recognition (OCR), Retrieval-Augmented Generation (RAG), and intelligent LaTeX-aware agents to manage the complex task of document synthesis.

The system is built around a multi-stage pipeline that includes:
1.  **Ingestion & OCR:** Processes input files (e.g., PDFs) and extracts text and layout information using an ensemble of OCR backends.
2.  **Chunking & Indexing:** Intelligently chunks the extracted content and builds a searchable vector index.
3.  **Document Planning:** Analyzes the document content to select an appropriate LaTeX document class (e.g., article, textbook, thesis) and plans the overall structure.
4.  **Knowledge Retrieval:** Queries a dedicated LaTeX knowledge base (KB) to retrieve necessary packages, commands, and environment templates.
5.  **LaTeX Synthesis:** Generates the final `.tex` file snippet by snippet, assembling the preamble, title, body content, and document-ending commands.
6.  **Compilation & Post-processing:** Compiles the generated LaTeX code and performs any necessary fix-ups to ensure a valid PDF output.

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
