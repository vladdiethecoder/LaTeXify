# Training Data Route & Filing Plan

This guide explains how to pull every dataset referenced in `training_database-catalog.json`, where to place the files, and how the resulting corpus maps to the LaTeXify training tracks (`T1`–`T8`). Use it as the single source of truth whenever you need to refresh the training data lake or onboard a new contributor.

## Folder Layout

```
training_data/
├── README.md                      # usage overview
├── route.json                     # auto-generated plan + track groupings
├── manifests/<slug>.json          # per-dataset ingest + preprocessing notes
├── raw/<slug>/.gitkeep            # staging area for downloaded assets
└── processed/<slug>/.gitkeep      # normalized/split output for modeling
```

All heavy assets live under `training_data/raw/<slug>` so we never mix third-party files with code. Downstream conversion (rendering, tokenization, templating) happens inside `training_data/processed/<slug>`.

## How to Build/Refresh the Route
1. **Generate manifests:** `make training-route` runs `scripts/build_training_data_route.py`, which parses `training_database-catalog.json`, writes `training_data/route.json`, and emits `training_data/manifests/<slug>.json` placeholders. The script also materializes empty directories for every dataset so filing stays consistent across machines.
2. **Review manifests:** Each manifest contains the ingest strategy (`git`, `huggingface-datasets`, or `manual`), the default pull command, known strengths/limitations, storage paths for each modality, and preprocessing reminders.
3. **Pull data:** `make prepare-training-data` runs `scripts/download_all_training_data.py`, which rebuilds manifests (if necessary) and then executes `scripts/prepare_training_data.py`. The helper loops through manifests, runs scripted git/HF commands (if the destination is empty), and automatically generates deterministic placeholder datasets for sources that cannot be fetched programmatically (DocVQA, FUNSD, IAM Handwriting, Shaip-linked corpora, etc.). The script uses the modern `hf download` command, so install the CLI via `pip install "huggingface_hub[cli]"`. If you later gain official access, replace the placeholders with the real archives in the same `raw/<slug>/source` folders.
4. **Normalize:** Convert inputs into model-ready form inside `training_data/processed/<slug>` (e.g., render page images, harmonize JSON schemas, build parquet/jsonl splits). The preparer script also writes `processed/<slug>/stats.json` with the current file counts so you can spot missing modalities quickly.

## Track-Level Coverage

| Track | Description | Datasets |
| --- | --- | --- |
| `T1` | Document layout + form understanding | DocLayNet, FUNSD |
| `T2` | Text detection in natural/scene imagery | Arabic Printed Text, Natural Environment OCR, SSIG-SegPlate, Street View House Numbers, TextOCR |
| `T3` | OCR & handwriting recognition | MathBridge, IAM Handwriting, Handwritten Chinese Characters, TextOCR, Street View House Numbers, SSIG-SegPlate, Natural Environment OCR, NIST, MNIST |
| `T4` | Table detection + structure recovery | DocLayNet, TableBank, Visual-TableQA |
| `T5` | Chart/diagram reasoning | Visual-TableQA |
| `T6` | Document QA / key-value extraction | DocVQA, FUNSD, Visual-TableQA |
| `T8` | Text-to-LaTeX / PDF infill | MathBridge |

## Per-Dataset Ingest Route

| Dataset | Tracks | Strategy | Default location / instructions |
| --- | --- | --- | --- |
| DocLayNet | T1, T4 | `git clone https://github.com/DS4SD/DocLayNet training_data/raw/doclaynet/source` | After cloning, copy PDFs to `raw/doclaynet/pdf`, render page images to `raw/doclaynet/images`, keep COCO JSON under `raw/doclaynet/annotations`. |
| TableBank | T4 | `git clone https://github.com/doc-analysis/TableBank training_data/raw/tablebank/source` | Contains rendered tables (Word/LaTeX). Preserve image splits inside `raw/tablebank/images` and export structured labels to `processed/tablebank`. |
| MathBridge | T3, T8 | `huggingface-cli download Kyudan/MathBridge --repo-type dataset --local-dir training_data/raw/mathbridge/source --local-dir-use-symlinks False` | Requires HF login. Use render service to materialize LaTeX into PDFs/images before moving curated splits to `processed/mathbridge`. |
| DocVQA | T6 | Manual (RRC portal) | Request credentials at https://www.docvqa.org/datasets, download each sub-task, and unpack into `raw/docvqa/source`. Mirror OCR JSON into `raw/docvqa/annotations`. |
| Visual-TableQA | T4, T5, T6 | `git clone https://github.com/AI-4-Everyone/Visual-TableQA training_data/raw/visual-tableqa/source` | Render LaTeX tables (`latex-src`) into images; align QA pairs with figures under `processed/visual-tableqa`. |
| FUNSD | T1, T6 | Manual | Reference the FUNSD release (linked via the Shaip article). Store scanned forms under `raw/funsd/images` and entity annotations under `raw/funsd/annotations`. |
| IAM Handwriting | T3 | Manual | Request IAM access (https://fki.tic.heia-fr.ch/databases/iam-handwriting-database). Keep word/line crops in `raw/iam-handwriting/images` and XML labels in `raw/iam-handwriting/annotations`. |
| Handwritten Chinese Characters | T3 | Manual | Acquire the CASIA-style dataset cited in the Shaip roundup, then arrange character bitmaps in `raw/handwritten-chinese-characters/images`. |
| TextOCR | T2, T3 | Manual (download from https://textvqa.org/datasets) | Move raw photos to `raw/textocr/images` and per-word annotations to `raw/textocr/annotations`. |
| Street View House Numbers | T2, T3 | Manual (`http://ufldl.stanford.edu/housenumbers/`) | Place `.mat` files / cropped digits under `raw/street-view-house-numbers/source` and export PNGs to `raw/.../images`. |
| NIST Database | T3 | Manual (`https://www.nist.gov/srd/nist-special-database-19`) | The SD19 release needs a request; once approved, unpack into `raw/nist-database/images`. |
| MNIST Database | T3 | Manual (`http://yann.lecun.com/exdb/mnist/`) | Store original idx files in `raw/mnist-database/source` and cache normalized tensors under `processed/mnist-database`. |
| Arabic Printed Text | T2 | Manual | Use the fonts/lexicon referenced in the catalog to synthesize samples under `raw/arabic-printed-text/text` and rasterize printable pages to `raw/.../images`. |
| SSIG-SegPlate | T2, T3 | Manual (`http://www.ssig.dcc.ufmg.br/segplate/`) | Keep cropped plates in `raw/ssig-segplate/images` and segmentation masks in `processed/ssig-segplate`. |
| Natural Environment OCR | T2, T3 | Manual | Follow the Shaip pointers to locate the dataset; preserve natural-scene crops in `raw/natural-environment-ocr/images` alongside transcripts. |

> 💡 Tip: Every manifest repeats the preprocessing hints from the catalog so you can script ETL pipelines directly from JSON (e.g., a CLI that loops over manifests to render PDFs, standardize annotation schemas, or build parquet shards).

## Next Steps
- Extend `scripts/build_training_data_route.py` whenever the catalog gains new sources or when a dataset calls for a bespoke filing layout.
- Wire the manifests into CI so nightly jobs can verify that all expected folders exist and warn when a dataset is missing from disk.
- Layer lightweight stats (record counts, storage footprint) into the manifests to track drift as we refresh upstream releases.
- Fine-tune local LaTeX models directly from the staged corpora via `scripts/finetune_latex_lm.py` (see `training_data/README.md` for invocation examples and supported flags).
