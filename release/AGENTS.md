# Repository Guidelines

## Mission
Operate inside `release/`, keeping the streamlined Florence → InternVL → Nougat → pytesseract OCR stack, SCAN-style layout segmentation, multimodal retrieval, and iterative critique loop healthy. No feature flags—every agent run should exercise the full chain unless explicitly told otherwise.

## Environment Setup
- Always create a local virtualenv: `python -m venv .venv && source .venv/bin/activate`.
- Install requirements via `pip install -r release/requirements.txt` (ships `torch`, `torchvision`, `transformers`, `pdf2image`, `sentencepiece`).
- External tools: Poppler (`pdfimages`, `pdftoppm`), `tesseract`, `latexmk`/`tectonic`. Local weights (Florence-2, InternVL-3.5, Nougat) must live under `<repo>/models/ocr/`. Pipeline is validated on dual RTX 3090s (24 GB each, no NVLink) + 32 GB DDR5 but will fall back to CPU/Nougat automatically.
- Populate `release/reference_tex/` with curated, high-aesthetic `.tex` sources (textbooks, papers). The RAG builder scans this tree to supply exemplar LaTeX environments for SpecialistAgents; the release ships with seeded physics/mechanics examples so future agents can build on a baseline style.

## Development Workflow
1. **Inputs** – place test PDFs in `release/inputs/` or use `release/samples/sample.pdf`.
2. **Run** – `python run_release.py --pdf sample.pdf --skip-compile` to exercise the full Parsing→Planner→Layout→StructureGraph→Retrieval→LaTeXSynth→LLMRefiner→Assembly→Validation stack. Artifacts include `master_plan.json`, `plan.json`, `graph.json`, `retrieval.json`, `snippets.json`, `preamble.json`, `synthesis_gaps.json`, `metrics.json`, `validation.json`, `reward_trace.jsonl`, and `agent_metrics.json`. The exemplar cache now persists globally at `release/cache/rag_index.json`, and the LLM refiner stores downloaded weights under `models/llm/`.
3. **Benchmark** – For READOC-style sweeps run `python run_release.py --benchmark-dir ../datasets/arxiv_subset --benchmark-limit 10 --skip-compile`; summaries land in `release/outputs/benchmark_summary.json`.
3. **Tests** – run `PYTHONPATH=. pytest -q release/tests` (covers ingestion/layout/synthesis contracts). End-to-end smoke: `PYTHONPATH=. pytest -q release/tests/test_smoke_release.py`. Static checks: `ruff check release release/scripts` and `mypy release release/scripts`. Add fixtures under `release/tests/data/` if needed. For CI, pass `--llm-mode off` (or export `LATEXIFY_LLM_MODE=off`) to skip heavy model loads.
4. **Docs & Metrics** – update `release/README.md` / `release/AGENTS.md` whenever you alter CLI flags, metadata schemas, or metric calculations.
5. **Artifacts** – never commit generated PDFs. Persist intermediate data only in `release/outputs/` and extend `.gitignore` if you introduce new cache paths.

## Coding Expectations
- Keep modules ~200–300 lines; add dedicated helpers for new responsibilities (retrieval, metrics, adapters, agents).
- Follow PEP 8 + type hints. Emit INFO logs when an agent starts/completes so latency tracking stays accurate.
- Preserve/extend structured metadata: `region_type`, `header_level`, `table_signature`, `list_depth`, OCR-noise scores, graph context, embeddings. Downstream consumers (RetrievalAgent, LaTeXSynthAgent, ValidationAgent) rely on these keys.
- Do not add ad-hoc CLI knobs; improvements should slot into the default cascade (Florence→InternVL→Nougat→pytesseract) and flow through the shared metadata contracts.

## Validation, Metrics & Resource Costing
- Inspect `graph.json`, `retrieval.json`, `snippets.json`, `main.tex`, `validation.json`, and `metrics.json`. Confirm structural gaps are fixed; log remaining issues in PR descriptions.
- Track section/table/list/equation fidelity, retrieval grounding, OCR-noise by region, compile success rate, and latency from `metrics.json` + `agent_metrics.json`.
- When changing OCR or chunking, compare noise distributions and manifest `ocr_usage` stats to keep Florence/InternVL loads predictable.
- Prefer deterministic smoke tests (use `release/samples/sample.pdf`) so future agents can reproduce results without external dependencies.

## Agent Workflow Blueprint

### Phase 1: Semantic Chunking
Fixed-size chunking destroys the logical flow of chronologically captured artifacts. The PlannerAgent must split documents by meaning, not token count. Iterate through the text, embed adjacent sentences (or tight spans), and compute cosine distance between sequential embeddings. Sharp distance spikes mark semantic breakpoints; the text between breakpoints becomes a coherent chunk that typically maps to a section or subsection downstream.

### Phase 2: Structured Outline Generation
Once chunks exist, the PlannerAgent (implemented in `planner.py`) calls an LLM constrained by a JSON schema/Pydantic model. The release build enforces this contract locally via Pydantic to keep `master_plan.json` machine-readable even when using heuristic fallbacks.

```json
{
  "document_title": "Vertical SHM and Angular Oscillation",
  "document_class": "article",
  "class_options": "12pt, twocolumn",
  "sections": [
    {
      "section_id": "sec-1",
      "title": "Vertical SHM",
      "content": [...]
    },
    {
      "section_id": "sec-2",
      "title": "Angular SHM",
      "content": [...]
    }
  ]
}
```

This “master plan” becomes the shared state object the entire workflow reads from and writes to.

### SpecialistAgent Fleet
Implemented under `specialists.py`, the orchestrator dispatches plan blocks to tightly scoped generators, then hands their output to the Qwen-based refiner for polishing (unless `--llm-mode off`):
- **TextAgent** normalizes paragraph chunks.
- **EquationAgent** produces amsmath environments and registers `amsmath`.
- **TableAgent** emits booktabs tables and requests `booktabs`.
- **ListAgent** and **FigureAgent** format structured lists or figure placeholders when assets are missing.
- **PreambleAgent** receives package requests from every specialist, deduplicates them, enforces ordering (e.g., `hyperref` last), and writes `preamble.json` alongside `snippets.json`.

Each specialist returns a LaTeX snippet keyed by `task_id`, guaranteeing deterministic assembly and dependency tracking.

### SynthesizerAgent
This deterministic Python node stitches the final candidate:
1. Load the PlannerAgent JSON.
2. Pull registered packages from the PreambleAgent.
3. Fetch the snippet map from all specialists.
4. Emit `\documentclass[...]`, the preamble, `\begin{document}`, ordered sections/subsections, and `\end{document}`.

Write the assembled `main.tex` into a temp build dir and surface its path to the validation loop.

### CompileAndRepairAgent
This stateful loop mirrors human LaTeX authoring:
1. **Execute** – run `pdflatex -interaction=nonstopmode -file-line-error -halt-on-error main.tex` via `subprocess.run`.
2. **Check** – inspect `returncode`. On success, exit and flag artifacts as validated.
3. **Reflect** – on failure, parse `main.log` with a purpose-built tool (`texoutparse`, `pydflatex`, or `texfot`) to extract structured errors (message, line, context).
4. **Repair** – prompt a repair LLM with the structured error:

    ```
    You are an expert LaTeX debugger. The following code in `main.tex` produced a compilation error.

    ERROR: ! Undefined control sequence.
    LINE: l.91
    OFFENDING_CODE_CONTEXT:
    ... l.90 \begin{figure}
    l.91 \incudegraphics[width=\linewidth]{img.png}
    l.92 \caption{A figure} ...

    Analyze the error and provide ONLY the corrected code wrapped in <latex_fix>...</latex_fix>.
    ```

   Parse the `<latex_fix>` payload and patch `main.tex`. Loop until success or hitting `max_attempts`, then escalate for human review.

### LangGraph Recommendation
Use LangGraph to model the workflow:
- A central GraphState holds the master plan, snippets, `main.tex`, compile logs, and attempt counts.
- Nodes (Planner, Specialists, Preamble, Synthesizer, Compile, Repair) read/write state explicitly.
- The Compile node feeds a conditional edge: `returncode == 0` routes to END; failures route to Repair, forming the necessary cycle.

### Re-indexing the Gold Standard
Existing vector stores keyed on text paragraphs are insufficient. Re-process gold-standard `.tex` sources and index semantically complete environments (`table`, `figure`, `equation`, `align`, etc.). Each vector stores both the LaTeX snippet and rich metadata:

```json
{
  "text": "\\begin{tabular}{lcc} \\toprule ...",
  "doc_id": "textbook-physics-ch9.tex",
  "type": "table",
  "packages": ["booktabs", "graphicx"],
  "style": "two-column-span"
}
```

### RAG as a Tool for Specialists
`release/rag.py` parses `reference_tex/**/*.tex` into environment-level entries (`table`, `figure`, `equation`, `align`) with inferred `domain` tags and builds the shared cache at `release/cache/rag_index.json` (use `--rag-refresh` to rebuild, or `python release/scripts/rag_eval.py` to inspect coverage). Specialists query by type/description/domain to obtain stylistic exemplars before drafting snippets. TableAgent, EquationAgent, ListAgent, and FigureAgent leverage these exemplars to set alignments/env choices, add `% RAG reference` comments, and auto-request packages surfaced by the retrieved entries. Example TableAgent prompt:

```
You are a LaTeX expert specializing in academic typesetting.
Task: Generate compilable LaTeX for a 3-column table comparing particle mass and charge.
Use the following high-quality table exemplars for style guidance:

[EXAMPLE 1]
...

Now output the final LaTeX for the requested table, including booktabs formatting and required packages.
```

Grounding generation in curated exemplars enforces the target look-and-feel.

Specialists also receive `% context:` comments summarizing their parent section title + summary plus `% parent-section:` (graph label) so snippets remain grounded even when retrieval fails.

### Gold-Standard Dataset Preparation

#### Supervised Fine-Tuning (SFT)
De-structure gold-standard `.tex` files into prompts using `pylatexenc`/`TexSoup` (convert LaTeX to plain text). Pair each prompt with the original LaTeX response and store as JSONL:

```json
{"prompt": "Introduction to Angular Momentum...", "response": "\\documentclass{...}\\section{Introduction..."} 
```

#### Preference Dataset
Create `{"prompt": "...", "chosen": "...", "rejected": "..."}` triples where:
- `chosen` is the pristine LaTeX.
- `rejected` is a hard negative generated via a mutation engine (syntactic corruptions, aesthetic downgrades like `$...$` or `\hline`-heavy tables, or LLM-generated imperfect drafts).

### Reward Schema
Implemented in `reward.py`, every run emits `rewards.json` plus an append-only `reward_trace.jsonl` with:
- `R_syntax`: derived from `validation.json` (1.0 on success, -1.0 on failure).
- `R_semantic`: BERTScore via `bert-score` (falls back to lexical overlap when transformers are unavailable).
- `R_aesthetic`: either the deterministic heuristics or the multimodal stub (select via `--reward-mode {heuristic,mm}`) that rewards booktabs, amsmath, margin controls, figures, and hyperref usage.
Weights default to `{syntax: 0.5, semantic: 0.3, aesthetic: 0.2}`, matching the phase-5 blueprint, but can be tuned per experiment. Traces capture per-run components + reward mode, and feed PPO/DPO alignment without rerunning TeX compilations.

### Fine-Tuning Strategy

#### PPO vs. DPO
- PPO (RLHF) excels at exploration-heavy syntax learning but is expensive because it needs live reward calls (compile, render, VLM).
- DPO is simpler, operating on static preference datasets, but lacks exploration.

#### Hybrid Approach
1. **Offline DPO (Semantics + Aesthetics)**  
   - Score the preference dataset with the combined slow rewards (`R_semantic + R_aesthetic`).  
   - Run DPO to bake these preferences into the SFT weights, distilling style and fidelity without live VLM calls.
2. **Online PPO (Syntax Only)**  
   - Start from the DPO-tuned policy (or resume a previous checkpoint via `--resume`).  
  - Render each policy response to PDF using the curated LaTeX templates under `release/templates/ppo/`, run the full pipeline (including the LLM refiner) for rewards, and log reward trajectories (raw + EMA) to TensorBoard (`--logdir`).  
   - Emit checkpoints every `--checkpoint-every` steps under `--output` so long runs can resume after interruptions while the policy explores new LaTeX compositions.

This hybrid delivers semantic/aesthetic excellence via offline distillation while preserving PPO’s strength at syntax exploration without incurring the full tri-modal reward cost per step.
