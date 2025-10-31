# Phase 2 — Aggregator + Compile Loop

Assemble `main.tex` from `plan.json` and per-task snippets in `snippets/`, enforce the **preamble allow-list**, **deduplicate labels**, and optionally **compile** with `latexmk`.

**Allow-list**

- Classes: `lix_article`, `lix_textbook`
- Packages: `amsmath, amssymb, amsthm, mathtools, thmtools, cleveref, graphicx, booktabs, caption, siunitx, microtype, enumitem, geometry, hyperref`
- `biblatex` is added **only if citations are detected** (`\cite{...}`, etc.). If `refs.bib` exists, it is added automatically.

## Run (MSC-AGG-01)

```bash
# 0) (venv as before; no extra deps required)
# 1) Ensure you have at least one snippet; from earlier steps you likely have:
#    - plan.json   (from planner)
#    - snippets/T03.tex (from synth step)  → if not, create a tiny placeholder:
mkdir -p snippets
printf "\\section{Introduction}\n\\label{sec:T03-introduction}\nHello.\n" > snippets/T03.tex

# 2) Aggregate (no compile)
python -m scripts.aggregator --plan plan.json --snippets_dir snippets --out_dir build --no_compile

# 3) Inspect
sed -n '1,120p' build/main.tex

# 4) (Optional) Try compiling if latexmk is installed
python -m scripts.aggregator --plan plan.json --snippets_dir snippets --out_dir build
# If latexmk is missing or you want to skip, add --simulate or --no_compile

# 5) Evidence logs
sed -n '1,200p' evidence/aggregate.log.jsonl
