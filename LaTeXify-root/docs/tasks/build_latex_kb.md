# Phase 0 — Build the LaTeX Help KB

This creates `kb/latex/latex_docs.jsonl`, chunks, and FAISS indexes (`latex_docs.index`, `faiss.index`), using only local/open-source tools.

## Prereqs

- Python 3.11+ (3.13 OK), venv with FAISS
- No paid APIs
- Deterministic: seed 42, stable sort orders
- Optional: `jq` for inspecting JSONL

On Fedora, install TeX Live later if you want to compile PDFs with `latexmk`:
```bash
sudo dnf install latexmk texlive-scheme-medium   # or scheme-full
