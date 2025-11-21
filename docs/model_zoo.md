# Model Zoo

This document lists the specialized models ("Experts") used in the LaTeXify Gen 3.0 pipeline.

## Core Pipeline Models

| Component | Model Name | Type | VRAM (FP16) | License | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Layout Analysis** | `DocLayout-YOLO` | YOLOv10 | ~500MB | AGPL-3.0 | Fine-tuned on DocStructBench. Detects Text, Title, Table, Figure, Formula. |
| **Math Recognition** | `UniMERNet` | Encoder-Decoder | ~1.3GB | Apache 2.0 | SOTA formula recognition. Handles nested LaTeX structures better than general VLMs. |
| **Table Structure** | `StructEqTable` | Transformer | ~2GB | Apache 2.0 | Reconstructs complex tables into LaTeX/HTML structure tokens. |
| **Text Extraction** | `PaddleOCR v4` | CNN+RNN | ~300MB | Apache 2.0 | High-speed multilingual OCR for standard typed text. |
| **Handwriting/Figure** | `Qwen2.5-VL-72B` | VLM | ~80GB (FP8) | Apache 2.0 | Used for handwriting, figure captioning, and complex layout reasoning. |
| **Refinement** | `Qwen2.5-Coder-32B` | LLM | ~20GB (FP8) | Apache 2.0 | Fixes LaTeX syntax errors, merges paragraphs, and ensures compilation. |

## Hardware Requirements & Configuration

### RTX 5090 (Recommended)
*   **Backbone**: `Qwen2.5-VL-72B` (Quantized to FP8)
*   **VRAM Usage**: ~45-60 GB depending on batch size.
*   **Optimizations**: FlashAttention-3, Torch Compile enabled.

### RTX 4090 / A100 (Minimal)
*   **Backbone**: `Qwen2.5-VL-7B` or `InternVL2-8B`
*   **VRAM Usage**: ~22 GB
*   **Trade-offs**: Lower reasoning capability for complex figures; slightly worse handwriting recognition.

### CPU Only (Development)
*   **Backbone**: Not supported for Vision. Refinement uses quantized CPU-offloaded models.
*   **Performance**: Very slow (not recommended for production).

## Model Sources & Weights

All models are downloaded automatically via Hugging Face Hub.

*   **UniMERNet**: `wanderkid/unimernet_base`
*   **DocLayout-YOLO**: `mashiro/DocLayout-YOLO` (Custom finetune may apply)
*   **Qwen2.5-VL**: `Qwen/Qwen2.5-VL-72B-Instruct`

## Quantization Strategy

Phase 4 utilizes **FP8 (Floating Point 8)** quantization for the large backbone models on RTX 5090/Blackwell architecture.

*   **Weights**: Stored in FP8 (E4M3).
*   **Activation**: Dynamic FP8 casting during inference.
*   **Library**: `torchao` or `vllm` built-in quantization.

## Fine-Tuning

See [FINE_TUNE_GUIDE.md](FINE_TUNE_GUIDE.md) for instructions on adapting these models to your specific domain data.
