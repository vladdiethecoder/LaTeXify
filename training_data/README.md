# Training Data Route

The `training_data/` directory stores manifests, placeholder directories, and downstream artifacts for all datasets referenced in `training_database-catalog.json`. Nothing is downloaded by default — the `scripts/build_training_data_route.py` utility materializes the folder structure and generates a manifest per dataset so that raw assets can be staged consistently before any heavy pulls occur.

## Usage
1. Ensure the Hugging Face CLI (`hf`) is installed (`pip install --upgrade "huggingface_hub[cli]"`) and you have the proper credentials for gated datasets via `huggingface-cli login` (the new `hf` command reuses that token).
2. Run `make training-route` (or `python scripts/build_training_data_route.py`) to:
   - Create `raw/<slug>` and `processed/<slug>` placeholders, one per dataset.
   - Emit `manifests/<slug>.json` with ingest strategy, pull commands, and preprocessing hints.
   - Update `route.json`, which summarizes the plan and groups datasets by track (`T1`–`T8`).
3. Pull + stage data via `make prepare-training-data` (one-click wrapper that rebuilds the route and runs all downloads). It loops over every manifest, runs scripted ingest commands when available, generates deterministic placeholder datasets automatically when upstream data is gated, and writes basic stats into `processed/<slug>/stats.json`.
4. Inspect each manifest and `processed/<slug>/stats.json` to verify counts, then normalize/convert assets into model-ready form under `training_data/processed/<slug>` (train/val/test).

## Ignoring Large Artifacts
The repository ignores everything inside `raw/` and `processed/` except `.gitkeep` placeholders. This lets you keep terabytes of PDFs or page images locally without polluting git history while still versioning the manifests/route files.

## Local Preparation Checklist

Use the following sequence whenever you need to bootstrap or refresh the full training corpus on a workstation:

1. **Prerequisites**
   - Install Python 3.10+ and `huggingface_hub` (`pip install huggingface_hub`) so the helper scripts can run.
   - Install the modern Hugging Face CLI (`pip install --upgrade "huggingface_hub[cli]"`) and authenticate with `huggingface-cli login` (the `hf` command uses the same credential store). This is required for datasets such as `Kyudan/MathBridge`.
   - Ensure `git` is in your `PATH` for repository-based datasets (DocLayNet, TableBank, Visual-TableQA).

2. **Generate manifests and directories**
   ```bash
   make training-route
   ```
   This step is idempotent and may be repeated anytime the catalog changes.

3. **Execute all scripted pulls**
   ```bash
   make prepare-training-data
   ```
   This single command runs `scripts/download_all_training_data.py`, which rebuilds manifests (if needed) and immediately executes the dataset ingests. Git and Hugging Face datasets are cloned/downloaded automatically into `training_data/raw/<slug>/source`. For sources that cannot be fetched programmatically, the script now auto-synthesizes placeholder datasets (metadata + sample files) so downstream tooling always sees consistent inputs without manual steps. Replace these placeholders with real data later if you obtain access.
   - DocLayNet is now fetched from IBM's object storage (core + extra archives). Once the download completes the converter runs automatically, emitting LayoutLM-ready JSONL into `training_data/processed/doclaynet/splits/{train,val,test}/data.jsonl`. You can rerun the converter manually with `python scripts/build_layout_splits.py --slugs doclaynet --force`.
   - PubLayNet, DocBank, GROTOAP2, RVL-CDIP, LADaS (diachronic), TQA, and Newspaper Navigator are now wired into the helper: `make prepare-training-data` will download their official archives (Hugging Face snapshots, Azure blobs, IBM CDN, AWS Open Data) into `training_data/raw/<slug>/source`. Gated datasets (M6Doc, CDSSE) still print manual instructions instead of fabricating placeholders.

4. **Normalize assets**
   - For each dataset, inspect `training_data/manifests/<slug>.json` to understand required preprocessing (e.g., render PDFs ➜ PNG, harmonize COCO JSON, convert `.mat` files to images).
   - Store converted outputs under `training_data/processed/<slug>/splits/{train,val,test}` and update `stats.json` if you add new artifacts (the helper script rewrites counts every time it runs).

5. **Verify**
   - Spot-check `training_data/route.json` to confirm all datasets list an `ingest_strategy` and that none of the `raw_root` paths are missing.
   - Confirm `stats.json` per dataset shows non-zero counts for the expected modalities before triggering downstream training jobs.

Keeping a consistent layout and manifest-driven workflow makes it easier to automate ETL jobs, cache statistics, and reason about coverage for each LaTeXify training track.

## Fine-Tuning Local Models
Once raw assets are filed, run `scripts/finetune_latex_lm.py` to fine-tune a causal LM on any slug inside `training_data/raw/`:

```bash
python scripts/finetune_latex_lm.py \
  --dataset-slug mathbridge \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --json-target-field latex \
  --json-input-field prompt \
  --max-records 2000 \
  --output-dir training_runs/mathbridge-qwen
```

- The CLI scans `training_data/raw/<slug>` for `.tex/.txt/.md/.json/.jsonl` files (override via `--extensions`), extracts text or JSON fields, and streams the samples into Hugging Face `datasets` before invoking `transformers.Trainer`.
- Use `--dry-run` to validate dataset discovery quickly (`make finetune-latex-lm` runs this sanity check for MathBridge).
- Customize `--prompt-template` when providing both input/target fields so instruction fine-tuning matches your downstream agent format. All runs drop metadata into `<output>/finetune_metadata.json` so experiments stay reproducible.
- Layout-aware datasets (DocLayNet, FUNSD, etc.) can be converted into JSON/JSONL splits with `tokens`, `bboxes`, and `ner_tags`, then fine-tuned locally via `python scripts/train_layoutlm_token_classifier.py --train-file <...> --validation-file <...>` (full command in `docs/FINE_TUNE_GUIDE.md`).
- Once you have the real JSON/JSONL files, replace the placeholders via:
  ```bash
  python scripts/register_layout_dataset.py \
    --slug docbank \
    --train-file /path/to/docbank_train.jsonl \
    --val-file /path/to/docbank_val.jsonl \
    --replace-placeholders
  ```
  Repeat per dataset so the fine-tuning scripts pick up the actual splits.
- If you just need a stub to unblock the trainer, pass `placeholder` for any split (e.g., `--train-file placeholder`). The helper will write a minimal JSONL entry automatically.
