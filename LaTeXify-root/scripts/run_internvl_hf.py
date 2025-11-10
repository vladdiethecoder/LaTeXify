#!/usr/bin/env python3
"""Minimal Hugging Face + Accelerate runner for InternVL.

This helper loads the bundled InternVL-Chat checkpoints with `device_map="auto"`
and optional per-device `max_memory` caps, then runs a single prompt against one
or more images.  The script mirrors the OpenAI-style UX we expect in ingestion
while keeping the dependencies light (Transformers + Accelerate only).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from PIL import Image
    from accelerate import PartialState
    from transformers import AutoModel, AutoTokenizer, CLIPImageProcessor

LOGGER = logging.getLogger("run_internvl_hf")
DEFAULT_MODEL_DIR = Path("models/ocr/internvl-3.5-14b")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run InternVL locally via Hugging Face + Accelerate (device_map auto)."
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(DEFAULT_MODEL_DIR),
        help="Path or repo id for the InternVL checkpoint (default: %(default)s).",
    )
    parser.add_argument(
        "--image",
        dest="images",
        action="append",
        default=[],
        help="Path to an image; repeat the flag to send multiple images.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="<image>\nDescribe this document chunk precisely.",
        help="Prompt to send to the model (default references a single image).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum new tokens per response (default: %(default)s).",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="Device map hint passed to Transformers (default: %(default)s).",
    )
    parser.add_argument(
        "--max-memory",
        action="append",
        default=[],
        metavar="DEVICE=VALUE",
        help="Optional per-device memory cap (e.g., cuda:0=22GiB). Repeat for multiple devices.",
    )
    parser.add_argument(
        "--offload-folder",
        type=str,
        default=None,
        help="Directory used by Transformers for CPU/NVMe offload checkpoints.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "bf16", "float16", "fp16", "float32", "fp32"],
        help="Torch dtype for model weights (default: %(default)s).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. Values > 0 enable sampling (default: %(default)s).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p nucleus sampling parameter (default: %(default)s).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate that the model directory exists and exit without loading weights.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args(argv)


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s | %(message)s")


def validate_model_dir(model_dir: str) -> Path:
    path = Path(model_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Model directory '{model_dir}' not found. Pass a local path with downloaded weights."
        )
    expected = path / "config.json"
    if not expected.exists():
        raise FileNotFoundError(
            f"Model directory '{model_dir}' is missing config.json; run scripts/install_models.py first."
        )
    return path.resolve()


def parse_max_memory_flags(flags: Iterable[str]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for raw in flags:
        if "=" not in raw:
            raise ValueError(f"Invalid --max-memory value '{raw}'. Use DEVICE=VALUE (e.g., cuda:0=20GiB).")
        device, value = raw.split("=", 1)
        device = device.strip()
        value = value.strip()
        if not device or not value:
            raise ValueError(f"Invalid --max-memory value '{raw}'.")
        result[device] = value
    return result


def detect_default_max_memory(torch_module: "torch") -> Dict[str, str]:
    budgets: Dict[str, str] = {}
    if torch_module.cuda.is_available():
        for idx in range(torch_module.cuda.device_count()):
            props = torch_module.cuda.get_device_properties(idx)
            total_gb = int(props.total_memory / (1024**3))
            safe_gb = max(total_gb - 2, 1)
            budgets[f"cuda:{idx}"] = f"{safe_gb}GiB"
    budgets.setdefault("cpu", "96GiB")
    return budgets


def resolve_dtype(name: str, torch_module: "torch") -> "torch.dtype":
    name = name.lower()
    mapping = {
        "bfloat16": torch_module.bfloat16,
        "bf16": torch_module.bfloat16,
        "float16": torch_module.float16,
        "fp16": torch_module.float16,
        "float32": torch_module.float32,
        "fp32": torch_module.float32,
    }
    dtype = mapping.get(name)
    if dtype is None:
        raise ValueError(f"Unsupported dtype '{name}'.")
    return dtype


def prepare_images(
    image_paths: Sequence[str],
    processor: "CLIPImageProcessor",
    dtype: "torch.dtype",
    target_device: "torch.device",
    torch_module: "torch",
    pil_image_module: "Image",
) -> Tuple["torch.Tensor | None", List[int] | None]:
    if not image_paths:
        return None, None
    tensors: List["torch.Tensor"] = []
    patch_counts: List[int] = []
    for raw_path in image_paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"Image '{raw_path}' does not exist.")
        image = pil_image_module.open(path).convert("RGB").resize((448, 448))
        batch = processor(images=image, return_tensors="pt").pixel_values.to(dtype)
        tensors.append(batch)
        patch_counts.append(batch.shape[0])
    pixel_values = torch_module.cat(tensors, dim=0).to(target_device)
    return pixel_values, patch_counts


def import_runtime_deps():
    try:
        import torch  # type: ignore[import-untyped]
        from PIL import Image  # type: ignore[import-untyped]
        from accelerate import PartialState  # type: ignore[import-untyped]
        from transformers import AutoModel, AutoTokenizer, CLIPImageProcessor  # type: ignore[import-untyped]
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Missing dependency '{exc.name}'. Install the extras with `pip install -e .[ocr] accelerate transformers`."
        ) from exc
    return torch, Image, PartialState, AutoModel, AutoTokenizer, CLIPImageProcessor


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)

    try:
        resolved_model_dir = validate_model_dir(args.model_dir)
    except FileNotFoundError as exc:
        LOGGER.error("%s", exc)
        return 1

    if args.dry_run:
        LOGGER.info("Dry run success — model directory: %s", resolved_model_dir)
        return 0

    try:
        (
            torch_module,
            pil_image_module,
            PartialStateClass,
            AutoModelCls,
            AutoTokenizerCls,
            CLIPImageProcessorCls,
        ) = import_runtime_deps()
    except RuntimeError as exc:
        LOGGER.error("%s", exc)
        return 1

    dtype = resolve_dtype(args.dtype, torch_module)
    try:
        max_memory = (
            parse_max_memory_flags(args.max_memory)
            if args.max_memory
            else detect_default_max_memory(torch_module)
        )
    except ValueError as exc:
        LOGGER.error("%s", exc)
        return 1

    try:
        partial_state = PartialStateClass()
        LOGGER.debug(
            "Accelerate state — rank: %s / %s",
            partial_state.process_index,
            partial_state.num_processes,
        )
    except Exception:
        partial_state = None

    tokenizer = AutoTokenizerCls.from_pretrained(resolved_model_dir, trust_remote_code=True, use_fast=False)
    model = (
        AutoModelCls.from_pretrained(
            resolved_model_dir,
            torch_dtype=dtype,
            device_map=args.device_map,
            max_memory=max_memory,
            low_cpu_mem_usage=True,
            use_flash_attn=torch_module.cuda.is_available(),
            trust_remote_code=True,
            offload_folder=args.offload_folder,
        )
        .eval()
    )

    target_device = torch_module.device("cuda:0" if torch_module.cuda.is_available() else "cpu")
    processor = CLIPImageProcessorCls.from_pretrained(resolved_model_dir)
    pixel_values, patch_counts = prepare_images(
        args.images,
        processor,
        dtype,
        target_device,
        torch_module,
        pil_image_module,
    )

    if args.images and "<image>" not in args.prompt:
        LOGGER.warning("Prompt does not contain '<image>' tokens even though images were provided.")

    generation_config = dict(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.temperature > 0,
    )
    LOGGER.info("Sending request with %s image(s); device_map=%s", len(args.images), args.device_map)

    response = model.chat(
        tokenizer,
        pixel_values,
        args.prompt,
        generation_config=generation_config,
        num_patches_list=patch_counts,
    )

    print("\nAssistant:\n----------")
    print(response)
    return 0


if __name__ == "__main__":
    sys.exit(main())
