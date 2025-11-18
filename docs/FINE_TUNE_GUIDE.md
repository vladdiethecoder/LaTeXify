# Fine-Tuning Quick Recipes

Copy/paste commands for each training stage using the tooling that already ships with LaTeXify. All commands assume you have staged datasets via `make training-route && make prepare-training-data` and you are working inside `release/.venv`.

```bash
source release/.venv/bin/activate
```

## 1. Core Layout Backbone (DocLayNet / PubLayNet / DocBank / M⁶Doc / GROTOAP2)
Fine-tune the LayoutLMv3 backbone using the in-repo trainer. Each split directory should contain a JSON/JSONL file with entries that look like:

```json
{"id": "page_00001", "tokens": ["The", "quick", "brown", "fox"], "bboxes": [[0,0,50,10], ...], "ner_tags": ["B-TITLE", "I-TITLE", "O", "O"]}
```

Then run:

```bash
python scripts/train_layoutlm_token_classifier.py \
  --model_name_or_path microsoft/layoutlmv3-base \
  --train-file training_data/processed/doclaynet/splits/train/data.jsonl \
  --validation-file training_data/processed/doclaynet/splits/val/data.jsonl \
  --output-dir training_runs/layoutlmv3-doclaynet \
  --per-device-train-batch-size 2 \
  --per-device-eval-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --num-train-epochs 5 \
  --learning-rate 2e-5 \
  --logging-steps 50 \
  --save-steps 500 \
  --fp16
```

Swap the `train/val` files for PubLayNet, DocBank, M⁶Doc, or GROTOAP2 exports to keep the backbone balanced.

If you prepared the splits elsewhere, register them under `training_data/processed/<slug>/splits/` with:

```bash
python scripts/register_layout_dataset.py \
  --slug publaynet \
  --train-file /data/publaynet_train.jsonl \
  --val-file /data/publaynet_val.jsonl \
  --replace-placeholders
```

Repeat for each slug (DocBank, M⁶Doc, GROTOAP2, etc.) so the trainer picks up your real annotations instead of the auto-generated placeholders. Pass `placeholder` for any split if you just want a stub entry (e.g., `--train-file placeholder`) while you finish the real export.
After training, point the pipeline at your checkpoint by setting
`LATEXIFY_LAYOUTLM_MODEL` to the run directory (e.g., `training_runs/layoutlmv3-doclaynet`)
before invoking `python run_release.py ...`. LayoutLM inference stays on CPU by default;
set `LATEXIFY_LAYOUTLM_DEVICE=auto` or `cuda:0` only if you explicitly want the
document-structure model on GPU. CLIP verification also defaults to CPU via
`LATEXIFY_CLIP_DEVICE=cpu` so the RTX 5090 is reserved for OCR.

> **One-click helper.** Drop the real `train.jsonl`, `val.jsonl`, or any `*.source.jsonl`
> files inside `training_data/processed/<slug>/splits/<split>/`, then run:
> ```bash
> python scripts/one_click_release.py \
>   --pdf release/samples/sample.pdf \
>   --layout-model training_runs/layoutlmv3-doclaynet \
>   --skip-release
> ```
> The script copies every `*.jsonl` (except `data.jsonl`) onto the canonical
> `data.jsonl`, ensures the LayoutLM processor assets are saved, and can optionally
> launch `run_release.py` for you. If any split is still missing its real
> annotations the helper aborts with a per-split checklist—drop the genuine
> JSONL files in place and rerun. Run it without `--skip-release` once the splits are ready.

## 2. Specialist Layout Domains (CDSSE / FUNSD / RVL-CDIP / Newspaper Navigator / Diachronic / TQA)
Use the same trainer but point it at specialist splits (FUNSD, RVL-CDIP, Newspaper Navigator, CDSSE, Diachronic, TQA). Example for FUNSD-style forms:

```bash
python scripts/train_layoutlm_token_classifier.py \
  --model_name_or_path microsoft/layoutlmv3-base \
  --train-file training_data/processed/funsd/splits/train/data.jsonl \
  --validation-file training_data/processed/funsd/splits/val/data.jsonl \
  --output-dir training_runs/layoutlmv3-funsd \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 16 \
  --num-train-epochs 15 \
  --learning-rate 1e-5 \
  --logging-steps 25 \
  --save-steps 250 \
  --fp16
```

For RVL-CDIP or Newspaper Navigator, convert the route manifest into token-classification JSON first (e.g., via a simple Python preprocessor) and reuse the same command. TQA can be preprocessed into LayoutLM inputs by rasterizing the PDF page, reading the ALTO/JSON metadata, and pointing `run_funsd` at that folder.

## 3. Tables & Math (TableBank / PubTabNet / PubTables-1M / SciTSR / FinTabNet / CROHME / im2latex / UniMER / LaTeX-OCR)
Use Microsoft’s Table Transformer (DETR-based) example that ships in `transformers` for table detection/structure, and our `scripts/finetune_latex_lm.py` for LaTeX outputs.

### Table Transformer for structural parsing
```bash
python -m transformers.examples.pytorch.table-transformer.run_table_transformer \
  --model_name_or_path microsoft/table-transformer-structure-recognition \
  --output_dir training_runs/tabletransformer-pubtables \
  --train_annotations_file training_data/processed/pubtables-1m/splits/train/annotations.json \
  --validation_annotations_file training_data/processed/pubtables-1m/splits/val/annotations.json \
  --remove_unused_columns False \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 10 \
  --learning_rate 5e-5 \
  --save_steps 1000 \
  --fp16
```

Swap the annotation paths for TableBank, PubTabNet, SciTSR, or FinTabNet (all manifests describe where the JSON lives). For CROHME/im2latex/UniMER/LaTeX-OCR, render formula crops + labels and use the same script with single-column “tables” to extract structure, or jump directly to the LLM recipe below.

### LaTeX generator fine-tuning
Use the new CLI to align the refiner on paired formula/text datasets:

```bash
python scripts/finetune_latex_lm.py \
  --dataset-slug im2latex-100k \
  --model-name Qwen/Qwen2.5-1.5B-Instruct \
  --json-input-field source \
  --json-target-field latex \
  --prompt-template "### Input:\n{input}\n\n### Output:\n{target}\n" \
  --max-records 200000 \
  --batch-size 2 \
  --gradient-accumulation 16 \
  --epochs 1 \
  --learning-rate 2e-5 \
  --output-dir training_runs/im2latex-qwen \
  --fp16
```

Repeat for UniMER, LaTeX-OCR, SciTSR (after exporting HTML to JSON), or any dataset that offers paired text ⇄ LaTeX.

## 4. Style & Aesthetics (LaTeX textbooks, arXiv tarballs, CCpdf)
The same CLI handles large LaTeX/style corpora—just point it at the staged `training_data/raw/<slug>` folder.

```bash
python scripts/finetune_latex_lm.py \
  --dataset-slug latex-textbooks \
  --model-name Qwen/Qwen2.5-1.5B-Instruct \
  --extensions .tex .md \
  --max-files 1000 \
  --max-records 500000 \
  --batch-size 2 \
  --gradient-accumulation 16 \
  --epochs 1 \
  --learning-rate 1e-5 \
  --prompt-template "{target}" \
  --output-dir training_runs/latex-style-qwen \
  --fp16
```

To soak up style from arXiv tarballs or CCpdf, drop the archives under `training_data/raw/<slug>/source`, run `make prepare-training-data`, and reuse the command above with the new slug.

## Notes
- Always regenerate manifests/stats before training (`make prepare-training-data`) so splits and counts stay current.
- The Hugging Face example modules referenced above are installed automatically with `transformers`. Adjust batch sizes/epochs to match GPU memory (single RTX 5090 = 32 GB VRAM).
- Every run writes metadata into `training_runs/<name>/finetune_metadata.json` so you can track which dataset/model/hparams produced a checkpoint.
