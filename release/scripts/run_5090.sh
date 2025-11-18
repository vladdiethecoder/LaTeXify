#!/usr/bin/env bash
# One-click wrapper for running LaTeXify on a single RTX 5090 (32 GB) host.
# Enforces sequential heavy OCR, releases VLMs between pages, and pins
# auxiliary models to CPU so Florence/InternVL always have headroom.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNNER="$ROOT/run_release.py"
INPUT_DIR="$ROOT/release/inputs"
LOG_DIR="$ROOT/release/logs/gpu_samples"

usage() {
  cat <<'EOF'
Usage: release/scripts/run_5090.sh <pdf-path> [additional run_release.py args]

Examples:
  release/scripts/run_5090.sh release/inputs/sample.pdf
  release/scripts/run_5090.sh release/inputs/sample.pdf --title "Example"

The wrapper:
  - forces LATEXIFY_* defaults that keep only one heavy OCR model on the GPU
  - warns when other workloads already occupy the 5090
  - captures a simple nvidia-smi log (disable via RUN5090_DISABLE_GPU_LOG=1)
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

PDF_ARG="$1"
shift

if [[ -f "$PDF_ARG" ]]; then
  PDF_PATH="$(realpath "$PDF_ARG")"
elif [[ -f "$INPUT_DIR/$PDF_ARG" ]]; then
  PDF_PATH="$(realpath "$INPUT_DIR/$PDF_ARG")"
else
  echo "[run_5090] Unable to find PDF at '$PDF_ARG' or '$INPUT_DIR/$PDF_ARG'." >&2
  exit 1
fi

if [[ ! -f "$RUNNER" ]]; then
  echo "[run_5090] run_release.py not found at $RUNNER" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export LATEXIFY_OCR_GPU_PREF="${LATEXIFY_OCR_GPU_PREF:-0}"
export LATEXIFY_OCR_SEQUENTIAL="${LATEXIFY_OCR_SEQUENTIAL:-1}"
export LATEXIFY_OCR_RELEASE_MODE="${LATEXIFY_OCR_RELEASE_MODE:-page}"
export LATEXIFY_OCR_FORCE_HEAVY="${LATEXIFY_OCR_FORCE_HEAVY:-1}"
export LATEXIFY_OCR_KEEP_LIVE="${LATEXIFY_OCR_KEEP_LIVE:-0}"
export LATEXIFY_CLIP_DEVICE="${LATEXIFY_CLIP_DEVICE:-cpu}"
export LATEXIFY_LAYOUTLM_DEVICE="${LATEXIFY_LAYOUTLM_DEVICE:-cpu}"
export LATEXIFY_LLM_DEVICE="${LATEXIFY_LLM_DEVICE:-cpu}"
export LATEXIFY_LLM_MIN_FREE_GB="${LATEXIFY_LLM_MIN_FREE_GB:-16}"

if command -v nvidia-smi >/dev/null 2>&1; then
  mapfile -t GPU_SNAPSHOT < <(nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits)
  if [[ ${#GPU_SNAPSHOT[@]} -gt 0 ]]; then
    IFS=',' read -r gpu_idx gpu_name mem_used mem_total <<<"${GPU_SNAPSHOT[0]}"
    mem_used_trimmed="$(echo "$mem_used" | xargs)"
    if [[ "$mem_used_trimmed" =~ ^[0-9]+$ && "$mem_used_trimmed" -gt 1500 ]]; then
      echo "[run_5090] Warning: GPU ${gpu_idx} (${gpu_name}) already uses ${mem_used_trimmed} MiB." >&2
    fi
  fi
else
  echo "[run_5090] nvidia-smi not found; skipping GPU pre-check." >&2
fi

GPU_LOG_PID=""
GPU_LOG_FILE=""
cleanup() {
  if [[ -n "$GPU_LOG_PID" ]]; then
    kill "$GPU_LOG_PID" >/dev/null 2>&1 || true
    wait "$GPU_LOG_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

if command -v nvidia-smi >/dev/null 2>&1 && [[ "${RUN5090_DISABLE_GPU_LOG:-0}" != "1" ]]; then
  mkdir -p "$LOG_DIR"
  stamp="$(date +"%Y%m%d_%H%M%S")"
  pdf_slug="$(basename "$PDF_PATH" | tr ' ' '_' | sed 's/[^A-Za-z0-9._-]/_/g')"
  GPU_LOG_FILE="$LOG_DIR/${stamp}_${pdf_slug}.csv"
  {
    echo "# timestamp, util%, mem_used(MiB), mem_free(MiB)"
    nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.free \
      --format=csv -l 2
  } > "$GPU_LOG_FILE" &
  GPU_LOG_PID=$!
  echo "[run_5090] Logging 5090 usage to $GPU_LOG_FILE (Ctrl+C will stop the sampler)." >&2
fi

echo "[run_5090] Running on PDF: $PDF_PATH" >&2
echo "[run_5090] Extra run_release.py args: $*" >&2

python -u "$RUNNER" --pdf "$PDF_PATH" "$@"

if [[ -n "$GPU_LOG_FILE" ]]; then
  echo "[run_5090] GPU usage log saved to $GPU_LOG_FILE" >&2
fi
