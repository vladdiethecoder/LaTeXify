# HOW_TO_INSTALL

Use this guide to provision the local model zoo under `models/`. All downloads stay offline after the first run so the pipeline can operate without an internet connection.

## 1. Prerequisites

1. Activate the project virtual environment (see `HOW_TO_RUN.md`).
2. Install the Hugging Face CLI helpers if you have not already (login is required for the gated Qwen 2.5 judge weights):
   ```bash
   pip install huggingface_hub
   huggingface-cli login   # required for Qwen 2.5 downloads
   ```
3. Ensure you have enough disk space (≈180 GB for the default set listed below).

## 2. Automated Installer

Run the helper script to pull the default suite of models:

```bash
python scripts/install_models.py --models all
```

Key flags:

- `--list` – show the available model keys and exit.
- `--models <key ...>` – install only the specified entries (see table below).
- `--dry-run` – print the planned actions without downloading.
- `--force` – re-download even if the destination folder already exists.

## 3. Model Inventory

| Key                     | Destination                         | Source Repo                                    | Notes |
|-------------------------|-------------------------------------|-----------------------------------------------|-------|
| `layout/qwen2.5-vl-32b` | `models/layout/qwen2.5-vl-32b/`     | `Qwen/Qwen2.5-VL-7B-Instruct`                 | Vision-language planner served via vLLM (32 B reference, 7 B weights for development). |
| `judge/qwen2.5-72b-gguf`| `models/judge/qwen2.5-72b-gguf/`    | `Qwen/Qwen2.5-72B-Instruct-GGUF`              | llama.cpp-compatible judge (downloads the `*q4_k_m.gguf` shard). Requires a Hugging Face account with access to the Qwen 2.5 weights. |
| `ocr/internvl-3.5-14b`  | `models/ocr/internvl-3.5-14b/`     | `OpenGVLab/InternVL-Chat-V1-2`                | Multi-purpose vision OCR used for figure/caption parsing. |
| `ocr/florence-2-large`  | `models/ocr/florence-2-large/`     | `microsoft/Florence-2-large-ft`               | Region-aware OCR. Requires accepting the Microsoft Florence license. |
| `ocr/nougat-small`      | `models/ocr/nougat-small/`         | `facebook/nougat-small`                       | Math-aware LaTeX reconstructor. |
| `ocr/mineru-1.2b`       | `models/ocr/mineru-1.2b/`          | Manual (https://github.com/NiuTrans/MinerU)   | MinerU weights are distributed outside HF. The installer creates a placeholder with instructions. |

The installer writes an `install_manifest.json` inside each automatically downloaded directory so you can audit what was pulled. Manual entries such as MinerU produce a `MANUAL_DOWNLOAD.txt` with the upstream URL and expected file placement.

## 4. Manual MinerU Setup

1. Follow the instructions at [NiuTrans/MinerU](https://github.com/NiuTrans/MinerU) to obtain the 1.2 B checkpoint (HF or custom mirror).
2. Convert the weights if necessary (the repo provides scripts for PyTorch and ONNX formats).
3. Drop the final files into `models/ocr/mineru-1.2b/`. The pipeline expects the loader scripts to find the tokenizer/config there.

## 5. Verifying the Install

You can re-run the script in dry-run mode to confirm everything is in place:

```bash
python scripts/install_models.py --dry-run
```

Additionally, `tree models/` should show the structure outlined above. Once populated, the standard `run_local.py` workflow (see `HOW_TO_RUN.md`) will automatically reference these weights.
