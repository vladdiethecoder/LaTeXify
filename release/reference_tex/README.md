# Reference TeX Exemplars

Curated `.tex` files placed under this tree serve as stylistic exemplars for the
LLM refiner and the specialist agents. Organize them by **domain** so different
documents can opt into the style that fits their content (e.g., `math/`,
`finance/`, `cs/`, `bio/`).

```
release/reference_tex/
  math/
    euler_notes.tex
  finance/
    q4_report.tex
  default/
    baseline_style.tex
```

Guidelines:

- Each file should be a short, high-quality LaTeX excerpt (a page or less).
- Capture full environments (tables, aligned equations, figure floats, etc.) so
  the refiner can imitate layout decisions.
- Keep sensitive or proprietary content out of the repository.
- After adding files, rerun `python run_release.py ... --rag-refresh` to rebuild
  the cached RAG index so new exemplars are discoverable.
