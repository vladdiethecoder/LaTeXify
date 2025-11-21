# Model Zoo

| Component | Model Name | VRAM (FP16) | Speed (5090) | License | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Layout** | `yolov10n.pt` (or fine-tuned) | ~500MB | >100 FPS | AGPL-3.0 | Used for segmentation. |
| **Math** | `unimernet_base` | ~1.3GB | ~50 eq/s | Apache 2.0 | Standard UniMERNet. |
| **Text** | `PaddleOCR v4` | ~300MB | High | Apache 2.0 | Multilingual support. |
| **Refiner** | `Qwen/Qwen2.5-Coder-14B-Instruct` | ~28GB | ~40 tok/s | Apache 2.0 | Quantization recommended for <24GB cards. |

## Compatibility Matrix

| Library | Version | Status |
| :--- | :--- | :--- |
| PyTorch | 2.2+ | Required |
| CUDA | 12.x | Recommended |
| FlashAttention | 2.x/3.x | Strongly Recommended |
| xformers | 0.0.25+ | Optional (Memory Efficient) |
