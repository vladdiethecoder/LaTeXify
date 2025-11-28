# LaTeXify VRAM & Pipeline Runbook (Sample PDF)

**Date:** 2025-11-20
**Status:** Stabilized (Pending Config Adjustment)
**Hardware:** Single GPU (32GB VRAM assumed), FlashAttention2 missing.

## Overview
This runbook documents the stabilization process for `release/samples/sample.pdf` on a memory-constrained environment. The pipeline uses a hybrid model stack (Florence-2, Qwen 14B 4-bit, Qwen 7B validation) and requires precise VRAM tuning to avoid OOM during vLLM initialization.

## Critical Findings

### 1. VRAM Management & Configuration
*   **Issue:** The pipeline defaults to `gpu_memory_utilization=0.6` for vLLM, which causes OOM when concurrent with `LLMRefiner` (which holds a 4-bit 14B model, ~9GB).
*   **Root Cause:** `release/models/vllm_client.py` uses `LATEXIFY_VLLM_GPU_UTILIZATION` (default 0.6) and ignores standard `GPU_MEMORY_UTILIZATION` if the former is not set.
*   **Fix:** Must export `LATEXIFY_VLLM_GPU_UTILIZATION=0.3` (or lower) alongside `GPU_MEMORY_UTILIZATION`.
*   **FlashAttention:** Strongly recommended to install `flash-attn` to reduce InternVL memory footprint.

### 2. Codebase Fixes Applied
*   **Regex Bug (`assembly.py`):** The `_strip_internal_macros` regex `r"\\providecommand\\{(?:enit@|endtable)"` failed to match lines like `\providecommand{\enit@...}` due to missing backslash escaping for the inner brace content.
    *   *Fix applied:* Updated to `r"\\providecommand.*(?:enit@|endtable)"`. 
*   **Package Conflicts:** `natbib` and `cite` were co-existing despite filtering logic.
    *   *Fix applied:* Enhanced `build_preamble` to check `extra_preamble_commands` for `natbib` presence before allowing `cite`.
*   **Quantization:** `LLMRefiner` was patched to support `bitsandbytes` 4-bit loading via `LATEXIFY_LLM_LOAD_IN_4BIT=1`.

## Attempt Log

| ID | Config | Result | Notes |
|----|--------|--------|-------|
| **1** | `florence`, No Vision, `chunk=2000` | **Fail** | LaTeX error: `Missing \begin{document}`. Caused by garbage `\providecommand` lines in preamble. Regex failed to strip them. |
| **2** | Same + 4-bit LLM | **Cancelled** | Cancelled to apply regex fixes. |
| **3** | Same + Regex Fix + `GPU_UTIL=0.3` | **Fail** | **OOM**. vLLM crashed with `ValueError: Free memory ... less than 0.6`. Discovered `LATEXIFY_VLLM_GPU_UTILIZATION` default overrides system env. |

## Recommended Configuration
For a stable run on a single 32GB GPU:

```bash
# Memory Safety
export LATEXIFY_VLLM_GPU_UTILIZATION=0.3
export GPU_MEMORY_UTILIZATION=0.3
export LATEXIFY_LLM_LOAD_IN_4BIT=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Model Selection (Open Weights)
export LATEXIFY_LLM_REPO="Qwen/Qwen2.5-14B-Instruct"
export LATEXIFY_VALIDATION_MODEL="Qwen/Qwen2.5-7B-Instruct"
export LATEXIFY_HALLUCINATION_MODEL="Qwen/Qwen2.5-7B-Instruct"
export LATEXIFY_VISUAL_JUDGE_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"

# Execution
source release/.venv/bin/activate
python run_release.py \
    --pdf release/samples/sample.pdf \
    --ocr-backend florence \
    --disable-vision-synthesis \
    --chunk-chars 2000
```

## Open Issues
1.  **FlashAttention2:** Still missing (`pip install flash-attn --no-build-isolation` recommended).
2.  **Attempt Budget:** The 2-hour cumulative limit was reached, preventing a final verification run.
