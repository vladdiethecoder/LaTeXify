# Evaluation Utilities

The evaluation stack relies on curated "golden" snippets stored under `dev/eval/golden/`.  Each
entry provides a `reference.tex` and a `reference.pdf` file that serve as ground truth for text and
image regressions.  Generated snippets written to `build/snippets/<name>.tex` (and optional PDF
renders at the same stem) are aligned to these references by directory name.

Run `scripts/evaluate_run.py` after a pipeline execution to compute aggregate metrics:

```bash
python scripts/evaluate_run.py --build-dir build --golden-dir dev/eval/golden
```

The script reads `build/pipeline_summary.json` (emitted by `scripts/pipeline.py`) to pick up compile
status and run metadata, then writes a consolidated JSON report containing text similarity scores
(WER/CER/BLEU-ish/METEOR-ish) and DSSIM-based PDF comparisons for any snippet PDF outputs.  Reports
are saved both to `build/evaluation_summary.json` and the matching entry under `dev/runs/<run_id>/`.
