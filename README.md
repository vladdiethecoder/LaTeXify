# LaTeXify Gen 3.0

**Research-grade PDF → LaTeX conversion for STEM documents**

A modular Pipeline-of-Experts system optimized for RTX 5090, converting complex academic PDFs (with math, tables, figures) into high-fidelity, compilable LaTeX.

## Features

- **DocLayout-YOLO**: SOTA layout detection fine-tuned on DocStructBench
- **UniMERNet**: >92% accuracy on mathematical formulas
- **StructEqTable**: Table structure recognition with LaTeX output
- **Qwen2.5-VL (72B)**: Vision-language model for figure captioning and refinement
- **RTX 5090 Optimized**: FP8 quantization, FlashAttention-3, vLLM serving
- **Visual Regression Testing**: SSIM-based quality validation

## Architecture

Pipeline stages:

1. **Ingestion**: PyMuPDF at 200 DPI → page images
2. **Layout Detection**: DocLayout-YOLO → bounding boxes + categories
3. **Content Extraction**:
   - Text: PaddleOCR (typed) or Qwen2.5-VL (handwriting)
   - Math: UniMERNet with SymPy validation
   - Tables: StructEqTable with LLM repair
   - Figures: Qwen2.5-VL captioning
4. **Reading Order**: XY-Cut algorithm for multi-column layouts
5. **Assembly**: Merge blocks → initial LaTeX
6. **Refinement**: Qwen2.5-VL via vLLM for syntax repair
7. **Compilation**: Tectonic with self-correction loop (max 3 attempts)

## Quick Start (Docker)

### RTX 5090 Setup

```bash
# Build RTX 5090 optimized image
docker build -f docker/Dockerfile.rtx5090 -t latexify:rtx5090 .

# Start vLLM server (in one terminal)
docker run --gpus all -p 8000:8000 latexify:rtx5090 \
  vllm serve Qwen/Qwen2.5-VL-72B-Instruct \
  --quantization fp8 --gpu-memory-utilization 0.9

# Run pipeline (in another terminal)
docker run --gpus all -v $(pwd):/workspace latexify:rtx5090 \
  python run_latexify.py input.pdf
```

**Prerequisites:**
- Docker Engine 24.0+
- NVIDIA Container Toolkit
- RTX 5090 GPU (or 4090 with reduced batch sizes)

### 2. Local Development

If you have a compatible environment (Linux, Python 3.10/3.11, CUDA 12.x):

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies
uv sync

# Install pre-commit hooks
pre-commit install
```

## Usage

### Run Pipeline

```bash
# Inside Docker or venv
python run_latexify.py --pdf input.pdf
```

### Run Demo UI

```bash
streamlit run apps/ui/app.py
```

## Golden Set Verification

1.  **Populate Golden Set**:
    ```bash
    python scripts/download_golden_set.py
    ```
    This downloads sample PDFs from arXiv to `data/golden_set/`.

2.  **Run Verification**:
    ```bash
    python scripts/verify.py --run-pipeline
    ```

## Documentation

*   [Model Zoo](docs/MODEL_ZOO.md)
*   [Troubleshooting Guide](docs/TROUBLESHOOTING.md)
*   [Verification Protocol](docs/VERIFICATION_PROTOCOL.md)
