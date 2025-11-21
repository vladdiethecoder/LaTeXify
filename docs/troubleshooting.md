# Troubleshooting Guide

Common issues and solutions for running LaTeXify Gen 3.0.

## Installation & Environment

### 1. "No kernel image available" or CUDA Errors on RTX 5090
**Cause**: PyTorch stable (2.4/2.5) does not fully support Blackwell (Compute Capability 12.0).
**Solution**:
*   Use the provided Docker image: `latexify:rtx5090`.
*   Or install PyTorch Nightly: `pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128`

### 2. `ModuleNotFoundError: No module named 'flash_attn'`
**Cause**: FlashAttention requires compilation against your specific CUDA version.
**Solution**:
```bash
pip install flash-attn --no-build-isolation
```
*Note: Ensure `nvcc --version` matches your PyTorch CUDA version.*

### 3. `RuntimeError: CUDA error: out of memory`
**Cause**: The pipeline is trying to load too many models simultaneously.
**Solution**:
*   **Reduce Batch Size**: Edit `config/hardware/rtx5090.yaml` -> `max_batch_size: 16`.
*   **Enable Offloading**: Ensure `vLLM` is managing the large model, not loaded directly in the python script memory.
*   **Use FP8**: Verify `optimization.use_fp8: true` is set in config.

## Pipeline Errors

### 4. UniMERNet `RuntimeError: trunk_base not found`
**Cause**: Mismatch between config model name and Hugging Face repo structure.
**Solution**:
*   Delete local cache: `rm -rf ~/.cache/huggingface/hub/models--wanderkid--unimernet_base`
*   Re-run to trigger fresh download.

### 5. PDF Processing Hangs / Timeout
**Cause**: Tectonic compiler entering an infinite loop or VLM taking too long on a complex page.
**Solution**:
*   The script has a 1200s timeout.
*   Check `output/phase4_run/pipeline.log` to see which stage is stuck.
*   If stuck on "Refinement", try `LATEXIFY_DISABLE_VLLM=1` to force CPU fallback (slow but stable) for debugging.

### 6. "ImportError: libGL.so.1: cannot open shared object file"
**Cause**: Missing system libraries for OpenCV.
**Solution**:
```bash
apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0
```

## Quality Issues

### 7. Formulas are generated as images or gibberish
**Cause**: The region was classified as 'Figure' instead of 'Equation'.
**Solution**:
*   Adjust confidence threshold in `config/pipeline.yaml`.
*   Fine-tune the YOLO model on your specific document style.

### 8. LaTeX compilation fails with "File not found"
**Cause**: Missing LaTeX packages in the base Tectonic bundle.
**Solution**:
*   Tectonic automatically downloads packages. Ensure internet access is available.
*   If offline, you must pre-cache the Tectonic bundle.

## Getting Help

If you encounter an issue not listed here:
1.  Check the logs in `output/`
2.  Run with `debug_run.sh` to capture verbose output.
3.  Open an issue on GitHub with the `pipeline.log` attached.
