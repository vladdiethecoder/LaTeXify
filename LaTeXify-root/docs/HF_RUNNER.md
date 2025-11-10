# InternVL HF Runner Setup

This guide walks through installing the optional Hugging Face runner dependencies and using the `scripts/run_internvl_hf.py` helper to exercise InternVL outside of vLLM.

## Install Dependencies

The runner uses Transformers, Accelerate, Torch, and Pillow. Install them (plus the core LaTeXify package) with the new optional extra:

```bash
pip install -e '.[hf-runner]'
```

If you already have Torch installed with a custom CUDA build, pin it before installing the extra:

```bash
pip install --no-deps torch==<your version>
pip install -e '.[hf-runner]'
```

## Running the Script

1. Download the InternVL weights via `python scripts/install_models.py internvl`.
2. Execute the runner (dry-run first to verify paths):

```bash
python scripts/run_internvl_hf.py --dry-run
python scripts/run_internvl_hf.py \
  --image models/ocr/internvl-3.5-14b/examples/image1.jpg \
  --prompt '<image>\nDescribe the figure.' \
  --max-new-tokens 256
```

To wire it into ingestion, switch the InternVL mode to HF and point at the script:

```bash
INTERNVL_MODE=hf \
INTERNVL_HF_RUNNER=scripts/run_internvl_hf.py \
INTERNVL_HF_MODEL_DIR=models/ocr/internvl-3.5-14b \
python -m latexify.ingestion.ingest_pdf --pdf inputs/doc.pdf --run-dir dev/runs/doc
```

Key flags:
- `--device-map auto` (default) uses Accelerate to shard layers across visible GPUs.
- `--max-memory cuda:0=21GiB --max-memory cpu=96GiB` lets you cap VRAM per device.
- `--image` can be repeated for multi-image prompts; remember to include `<image>` placeholders in the prompt.
- `--offload-folder /tmp/internvl-offload` stores weights/KV overflow on NVMe without changing code.

## Offload & Memory Notes

- When GPUs are tight, combine `--device-map auto` with per-device `--max-memory` so Accelerate drops layers onto CPU. The script auto-detects a conservative budget if you don’t supply flags.
- To offload even more, pair `--max-memory cpu=128GiB` with OS-level NVMe swap or `torch.distributed.fsdp_cpu_offload`. Monitor throughput—CPU hops can increase latency per chunk.
- Keep `bfloat16` as the default dtype; fall back to `float16` on cards that lack BF16 (e.g., RTX 20-series).
- “Knows about both 3090s”: set `INTERNVL_HF_GPUS="0,1"` (or `gpus: "0,1"` in `configs/ingestion/internvl.yaml`) and `INTERNVL_HF_MAX_MEMORY="cuda:0=21GiB,cuda:1=21GiB,cpu=128GiB"` so the runner auto-shards layers while spilling overflow to CPU/NVMe when needed.
- If you need strict isolation, place the runner on a separate host/process and refer to it via `INTERNVL_HF_RUNNER="ssh user@runner python scripts/run_internvl_hf.py"` or a supervisor unit (see `configs/ingestion/internvl.yaml` for a systemd-ready example).

## Configuration & Environment

- Edit `configs/ingestion/internvl.yaml` to pin the HF runner mode, runner script, CUDA devices (e.g., both 3090s via `gpus: "0,1"`), per-device `max_memory`, `offload_folder`, and subprocess timeout. Point ingestion at a different file with `INTERNVL_CONFIG=/path/to/custom.yaml`.
- Runtime overrides do not require code changes—export env vars to tweak behavior on the fly:
  - `INTERNVL_MODE` (`vllm` or `hf`) toggles the backend.
  - `INTERNVL_HF_RUNNER`, `INTERNVL_HF_MODEL_DIR`, `INTERNVL_HF_PYTHON` switch binaries/weights.
  - `INTERNVL_HF_GPUS` (or `INTERNVL_HF_VISIBLE_GPUS`) sets `CUDA_VISIBLE_DEVICES`.
  - `INTERNVL_HF_DEVICE_MAP`, `INTERNVL_HF_MAX_MEMORY="cuda:0=21GiB,cpu=96GiB"`, and `INTERNVL_HF_OFFLOAD_FOLDER` flow directly into the runner CLI.
- `INTERNVL_HF_TIMEOUT` caps subprocess runtime in seconds; `INTERNVL_HF_RETRIES` controls how many times we retry on failures; `INTERNVL_HF_EXTRA_ARGS` appends arbitrary runner flags.
- Want everything wired automatically? `python scripts/setup_one_click.py --pdf '<sample>'` installs dependencies (inside an auto-created virtualenv using a supported Python 3.10–3.13 interpreter), downloads models, launches vLLM endpoints, runs the HF runner dry-run, and executes a pipeline/pytest smoke test (see README for flags).

## Troubleshooting

- `ModuleNotFoundError`: ensure the `hf-runner` extra is installed inside the same virtualenv you are using to launch the script.
- `CUDA out of memory`: lower `--max-new-tokens`, add more aggressive `--max-memory` caps, or export `CUDA_VISIBLE_DEVICES` to limit placement to a larger GPU.
- `ValueError: prompt missing <image>`: include at least one `<image>` tag per uploaded image; the model expects that token to anchor the visual context.
- Runner fails but vLLM succeeds? Set `INTERNVL_MODE=vllm` (or simply unset it) to fall back to the existing HTTP path. This keeps the old deployment ready while you validate HF mode.
