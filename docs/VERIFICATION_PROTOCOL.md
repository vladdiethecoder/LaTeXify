# Golden Set Verification Protocol

## Overview

The Golden Set verification is the primary quality gate for the LaTeXify pipeline. It moves evaluation from qualitative ("it looks good") to quantitative ("94% BLEU score").

## The Golden Set

The Golden Set consists of 50 diverse STEM PDFs sourced from arXiv, covering:
*   Mathematics (heavy equation usage)
*   Computer Science (code blocks, algorithms)
*   Physics (complex notation, diagrams)
*   Biology/Chemistry (tables, chemical formulas)

**Location:** `data/golden_set/`

Each sample comprises:
1.  `{sample_id}.pdf`: The input document.
2.  `{sample_id}.tex`: The ground-truth LaTeX source (cleaned of comments and preamble macros).

## Methodology

We use **BLEU (Bilingual Evaluation Understudy)** score to measure the similarity between the generated LaTeX and the ground truth. While typically used for translation, it serves as a robust proxy for OCR and structural accuracy in LaTeX.

### Metric: BLEU-4

*   **Tokenizer**: Whitespace-based tokenization (LaTeX commands are treated as words).
*   **Smoothing**: Method 4 (Chen & Cherry, 2014) to handle short segments.

### Thresholds

| Level | BLEU Score | Status |
| :--- | :--- | :--- |
| **Excellent** | > 0.90 | Production ready. |
| **Good** | 0.80 - 0.90 | Minor formatting differences. |
| **Acceptable** | 0.70 - 0.80 | Readable, but requires human review. |
| **Fail** | < 0.70 | Structural failures or significant OCR errors. |

## Running Verification

The `scripts/verify.py` tool automates this process.

```bash
# 1. Dry Run (check for missing files)
python scripts/verify.py

# 2. Full Verification (runs pipeline)
python scripts/verify.py --run-pipeline
```

### Output

The script produces:
*   Per-file BLEU scores.
*   Average dataset BLEU score.
*   (Planned) HTML diff report.

## Contributing to the Golden Set

To add a new sample:
1.  Select a CC-BY licensed paper from arXiv.
2.  Download the PDF and Source.
3.  Flatten the Source into a single `.tex` file using `latexpand`.
4.  Place both in `data/golden_set/`.
5.  Commit the files.
