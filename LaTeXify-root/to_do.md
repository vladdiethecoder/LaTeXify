# LaTeXify Project: To-Do List

This file tracks the development, refinement, and future ambitions for the LaTeXify project.

## Priority Tasks / Next Steps

-   [ ] **Improve Synthesis Agent:** Enhance `synth_latex.py` to better handle complex layouts, figures, and tables.
-   [ ] **Final Compilation Loop:** Implement a robust `pdflatex` compilation loop with error detection and automated retries (e.g., fixing undefined references).
-   [ ] **Refine Document Planner:** Improve the `phase2_planner.md` logic to create more detailed and accurate document structure plans.
-   [ ] **End-to-End Evaluation:** Finalize the end-to-end evaluation metric (`dev/eval/metrics.py`) and run it against the "Basic Skills Review" test set.
-   [ ] **Fix LFS Issues:** The root `README.md` and `to_do.md` were previously stored as LFS pointers. Replace them with these new text files.

## Core Pipeline & Agents

-   [ ] **OCR Consensus:** Refine the `ocr_ensemble.py` logic. The current `consensus.py` script is a good start, but we need a more robust voting/merging mechanism for text blocks.
-   [ ] **Chunking Strategy:** Experiment with different chunking strategies (`scripts/chunk_strategies.py`). The current page-aware chunker (`dev/chunking/page_aware_chunker.py`) is a good baseline, but semantic chunking might be better.
-   [ ] **Document Class Selection:** Expand the `kb/classes/` directory to include more document types (e.g., `beamer` for presentations, `book`). Improve the `choose_doc_class.py` agent to be more accurate.
-   [ ] **Preamble Generation:** The `gate6_preamble.py` script needs to be more dynamic. It should select packages based on *retrieved* content needs (e.g., add `graphicx` if images are detected, `amsmath` for complex equations).
-   [ ] **Figure & Table Processing:** Develop a specialized agent for identifying figures and tables from OCR output and synthesizing the correct LaTeX `figure` or `table` environments.

## Knowledge Base (KB)

-   [ ] **Expand LaTeX KB:** Add more advanced topics to `kb/latex/latex_docs.jsonl` (e.g., TikZ, bibliography management with BibTeX/BibLaTeX, custom command definitions).
-   [ ] **Re-index KB:** Re-run `scripts/build_latex_kb.py` and `scripts/build_index.py` after expanding the KB.
-   [ ] **Improve Retrieval:** Fine-tune the retrieval agent (`scripts/retrieval_agent.py`) and experiment with rerankers (`scripts/rerankers.py`) to get more relevant snippets.

## Development & Ops

-   [ ] **CI/CD:** Set up a GitHub Actions workflow to run the test suite (`tests/`) on push.
-   [ ] **Dependency Management:** Clean up `dev/requirements-ocr.txt` and create a single `requirements.txt` for the whole project.
-   [ ] **Model Registry:** Formalize the `scripts/model_registry.py` to easily swap between different embedding, OCR, and generation models.
-   [ ] **Clean up `/dev/runs`:** Archive old test runs to keep the repository clean.
-   [ ] **Documentation:** Write detailed developer documentation (the current `dev/README.dev.md` appears to be an LFS pointer and should be replaced).

## Future Ambitions (v2.0)

-   [ ] **Web UI:** Create a simple web interface (e.g., Flask or Streamlit) to upload a PDF and get back a compiled `.tex` file and PDF.
-   [ ] **Support .docx and .md:** Add ingestion pipelines for Word documents and Markdown files, not just PDFs.
-   [ ] **Batch Processing:** Add support for processing entire directories of documents at once.
-   [ ] **Interactive Mode:** Create an "interactive" mode where the user can correct the LaTeX output in a live-preview environment.
