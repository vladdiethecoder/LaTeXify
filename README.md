# LaTeXify

A modular, high-fidelity pipeline for converting PDFs to clean, compilable LaTeX.
Optimized for RTX 5090 hardware with 32GB VRAM.

## Architecture

This system uses a multi-stage expert model approach:

1.  **Ingestion**: High-fidelity rasterization (300-400 DPI) using PyMuPDF.
2.  **Layout Analysis**: YOLOv10/RT-DETR based object detection (Text, Tables, Equations).
3.  **Expert Recognition**:
    *   **Math**: UniMERNet
    *   **Text**: PaddleOCR
    *   **Tables**: TableMaster / StructureMaster (Planned)
4.  **Assembly**: Topological sorting and reconstruction.
5.  **Refinement**: Local LLM (Qwen2.5-Coder) for syntax repair and glitch correction.

## Directory Structure

```
LaTeXify/
├── config/             # Hydra configuration
├── src/latexify/
│   ├── ingestion/      # PDF -> Image
│   ├── layout/         # YOLO Logic
│   ├── ocr/            # PaddleOCR Wrapper
│   ├── math/           # UniMERNet Wrapper
│   ├── refinement/     # LLM Refiner
│   └── core/           # Pipeline Logic
└── scripts/            # Utilities
```

## Setup

### 1. Docker (Recommended)

The easiest way to run LaTeXify with full GPU support (CUDA 12.4) is using Docker Compose.

```bash
# Build and Run Pipeline
docker compose up --build

# Run Interactive Shell
docker compose run --rm latexify /bin/bash
```

**Prerequisites:**
*   Docker Engine
*   NVIDIA Container Toolkit (`nvidia-docker2`)

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
