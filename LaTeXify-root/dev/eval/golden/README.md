# Golden References

This directory contains a curated suite of PDF snippets with their vetted LaTeX sources.  Each
subdirectory represents a logical evaluation target and must include:

* `reference.pdf` – the rendered artifact used for visual comparison.
* `reference.tex` – the ground-truth LaTeX snippet used for text-level metrics.

The references are consumed by `scripts/evaluate_run.py` when generating evaluation reports.  New
references can be added by creating another subdirectory following the same structure.  Make sure
that the `.tex` is normalized (no trailing whitespace, consistent math mode) so that string
comparisons and tokenization produce stable metrics.
