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





# LaTeXify Project: To-Do List (tracksheet)

This file tracks the project's priorities, focusing on robustness and building a self-improving system.

## Completed Goals

-   [x] **Core Pipeline Architecture:** The multi-agent "Plan, Retrieve, Synthesize, Aggregate" pipeline is implemented and functional.
-   [x] **Document Planner:** The `planner_scaffold.py` agent successfully creates `plan.json` blueprints.
-   [x] **RAG System:** The LaTeX KB (`/kb/latex/`) is indexed with FAISS, and the `retrieval_agent.py` creates data bundles.
-   [x] **Advanced Layout Analysis:** The `/pdf-document-layout-analysis/` submodule is integrated for sophisticated document parsing.
-   [x] **Multi-Agent Framework:** The `tasks/` directory and agent scripts (`synth_latex.py`, etc.) provide a "prompt-as-code" framework.
-   [x] **LFS Issue Resolution:** Replaced root LFS pointers with text files.

## Priority Tasks (NEED to be done)

-   [x] **Final Compilation Loop:** Implement a robust `pdflatex` compilation loop in the pipeline. It must detect errors (by parsing the `.log`) and identify the failing snippet.
-   [ ] **Synthesis Agent (Complex Content):** Enhance `synth_latex.py` to reliably handle complex layouts, multi-column text, figures (`graphicx`), and tables (`tabular`, `booktabs`).
-   [ ] **Integrate Evaluation Metric:** Finalize and integrate `dev/eval/metrics.py`. Automatically score all test runs in `/dev/runs/` against their ground truth.
-   [ ] **Dependency Management:** Consolidate `dev/requirements-ocr.txt` and other dependencies into a single root `requirements.txt`.

## Robustness & Accuracy (The "Self-Improving" Loop)

-   [x] **Self-Correction Agent:** This is the top priority for robustness.
    -   **Task:** If the compilation loop fails, a new 'correction' agent should be triggered.
    -   **Input:** The failing `.tex` snippet, the `pdflatex` error log, and the original data bundle.
    -   **Action:** The agent will read the error, query the LaTeX KB for solutions, and attempt to re-synthesize *only* the failing snippet.
-   [ ] **Auto-Expanding KB:** Create a feedback loop to make the system smarter over time.
    -   **Task:** When the `metrics.py` score is high *and* a human validator approves, or when a "self-correction" is successful, add the novel (e.g., a complex table) snippet and its solution back to the `data/latex_docs.jsonl` for re-indexing.
-   [ ] **Dynamic Preamble:** Make `gate6_preamble.py` dynamic. It should select packages based on the *actual* synthesized content (e.g., add `amsmath` only if `align` or `equation` environments are used).
-   [ ] **OCR Consensus:** Refine `scripts/consensus.py` to implement a robust voting/merging mechanism for the different OCR backends, improving text accuracy.
-   [ ] **Retrieval Reranking:** Implement `scripts/rerankers.py` in the `retrieval_agent.py` to improve the quality and relevance of RAG context.

## Future Ambitions (v2.0)

-   [ ] **Web UI:** Create a simple Flask/Streamlit interface to upload a PDF and receive a compiled `.tex` file and PDF.
-   [ ] **Multi-Format Ingestion:** Add support for `.docx` and `.md` files, not just PDFs.
-   [ ] **Interactive Mode:** A "human-in-the-loop" UI where a user can approve/reject/edit the planner's `plan.json` or correct a failing `.tex` snippet in a live-preview environment.
