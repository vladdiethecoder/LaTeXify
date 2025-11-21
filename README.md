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

1.  **Docker (Recommended)**:
    ```bash
    docker build -t latexify .
    docker run --gpus all -v $(pwd):/app latexify
    ```

2.  **Local Development**:
    ```bash
    uv sync
    pre-commit install
    ```

## Usage

Run the main pipeline:

```bash
python run_latexify.py --pdf input.pdf
```

Run the Demo UI:

```bash
streamlit run apps/ui/app.py
```

## Documentation

*   [Model Zoo](docs/MODEL_ZOO.md)
*   [Troubleshooting Guide](docs/TROUBLESHOOTING.md)
*   [Verification Protocol](docs/VERIFICATION_PROTOCOL.md)

## Quality Assurance

We verify pipeline accuracy using a 50-document Golden Set.

```bash
python scripts/verify.py
```