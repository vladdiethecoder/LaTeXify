# Troubleshooting Guide

## CUDA Ecosystem

| Error | Cause | Solution |
| :--- | :--- | :--- |
| `RuntimeError: CUDA error: out of memory` | VRAM exceeded (32GB limit) | Reduce `batch_size` in `config/pipeline.yaml`. Enable `load_in_8bit=True` for Refiner. |
| `ModuleNotFoundError: No module named 'flash_attn'` | FlashAttention not installed | Run `pip install flash-attn --no-build-isolation`. Ensure CUDA 12.4. |
| `RuntimeError: trunk_base not found` | UniMERNet model mismatch | Ensure `config/model/unimer.yaml` points to the correct HF repo. |

## Docker

*   **Permission Denied**: Run `docker` with `sudo` or add user to `docker` group.
*   **GPU not found**: Use `--gpus all` flag. Ensure NVIDIA Container Toolkit is installed.

## Pipeline

*   **Low Accuracy on Math**: Check dpi settings. Ensure `unimernet_base` is used, not `tiny`.
*   **Hallucinations**: Enable "N-Gram Verification" (Planned for v1.1).
