#!/usr/bin/env python3
"""Download and organize LaTeXify's local model dependencies."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional
import shutil

# ---------------------------------------------------------------------
# Robust Hugging Face import
# ---------------------------------------------------------------------
# Your environment has huggingface_hub 1.1.2, which does NOT export
# HfHubHttpError and may not export HfHubHTTPError at top level.
# This block works across those variants.
try:
    from huggingface_hub import snapshot_download  # type: ignore

    try:
        # try top-level first
        from huggingface_hub import HfHubHTTPError  # type: ignore
    except ImportError:
        try:
            # fall back to utils
            from huggingface_hub.utils import HfHubHTTPError  # type: ignore
        except ImportError:
            # last resort: define a dummy so code compiles
            class HfHubHTTPError(Exception):  # type: ignore
                pass

except Exception as e:  # pragma: no cover - show real import problem
    print("[install_models] failed to import huggingface_hub:", e, file=sys.stderr)
    snapshot_download = None  # type: ignore

    class HfHubHTTPError(Exception):  # type: ignore
        pass


REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    repo_id: Optional[str]
    target: Path
    allow: Optional[Iterable[str]] = None
    revision: Optional[str] = None
    manual_url: Optional[str] = None
    notes: Optional[str] = None

    def destination(self) -> Path:
        return MODELS_DIR / self.target


# ---------------------------------------------------------------------
# Model registry (fixed repo_id for layout model)
# ---------------------------------------------------------------------
MODEL_REGISTRY: Dict[str, ModelSpec] = {
    # FIXED: this was "Qwen/Qwen2.5-VL-32B" (404), now correct:
    "layout/qwen2.5-vl-32b": ModelSpec(
        key="layout/qwen2.5-vl-32b",
        repo_id="Qwen/Qwen2.5-VL-32B-Instruct",
        target=Path("layout/qwen2.5-vl-32b"),
        notes="Vision-language layout model for PDF/page understanding.",
    ),
    "judge/qwen2.5-72b-gguf": ModelSpec(
        key="judge/qwen2.5-72b-gguf",
        repo_id="Qwen/Qwen2.5-72B-Instruct-GGUF",
        target=Path("judge/qwen2.5-72b-gguf"),
        allow=["*q4_k_m.gguf"],
        notes="llama.cpp-compatible judge (Q4_K_M). Requires accepting the Qwen 2.5 license on Hugging Face.",
    ),
    "ocr/internvl-3.5-14b": ModelSpec(
        key="ocr/internvl-3.5-14b",
        repo_id="OpenGVLab/InternVL-Chat-V1-2",
        target=Path("ocr/internvl-3.5-14b"),
        notes="InternVL vision OCR (chat tuned).",
    ),
    "ocr/florence-2-large": ModelSpec(
        key="ocr/florence-2-large",
        repo_id="microsoft/Florence-2-large-ft",
        target=Path("ocr/florence-2-large"),
        notes="Florence-2 Large fine-tuned for OCR/regional text.",
    ),
    "ocr/nougat-small": ModelSpec(
        key="ocr/nougat-small",
        repo_id="facebook/nougat-small",
        target=Path("ocr/nougat-small"),
        notes="Nougat LaTeX OCR specialist.",
    ),
    "ocr/mineru-1.2b": ModelSpec(
        key="ocr/mineru-1.2b",
        repo_id=None,
        target=Path("ocr/mineru-1.2b"),
        manual_url="https://github.com/NiuTrans/MinerU",
        notes="Manual download required (NiuTrans MinerU weights).",
    ),
}


def ensure_hf_available() -> None:
    if snapshot_download is None:
        raise RuntimeError(
            "huggingface_hub is required for automatic downloads. "
            "Install via `pip install huggingface_hub` inside your venv."
        )


def list_models() -> None:
    print("Available models:")
    for key, spec in MODEL_REGISTRY.items():
        print(f"- {key}")
        if spec.repo_id:
            print(f"  repo:   {spec.repo_id}")
        print(f"  target: {spec.destination()}")
        if spec.manual_url:
            print(f"  manual: {spec.manual_url}")
        if spec.notes:
            print(f"  notes:  {spec.notes}")
        print()


def install_model(spec: ModelSpec, *, dry_run: bool = False, force: bool = False) -> None:
    dest = spec.destination()

    # manual model
    if spec.repo_id is None:
        dest.mkdir(parents=True, exist_ok=True)
        manual_file = dest / "MANUAL_DOWNLOAD.txt"
        if not manual_file.exists() or force:
            manual_file.write_text(
                f"This model must be downloaded manually.\n"
                f"URL: {spec.manual_url or 'N/A'}\n"
                f"Notes: {spec.notes or 'N/A'}\n"
            )
        print(f"[{spec.key}] Manual install placeholder written -> {manual_file}")
        return

    dest_has_payload = dest.exists() and any(dest.iterdir())
    if dry_run:
        if dest_has_payload:
            print(f"[dry-run] Would skip {spec.key}; already present at {dest}")
        else:
            print(f"[dry-run] Would download {spec.repo_id} -> {dest}")
        return

    ensure_hf_available()

    if dest_has_payload and not force:
        print(f"[{spec.key}] already exists at {dest} (use --force to re-download)")
        return

    if dest.exists() and force:
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {spec.repo_id} to {dest} ...")

    kwargs = {
        "repo_id": spec.repo_id,
        "local_dir": str(dest),
        "revision": spec.revision,
        "allow_patterns": list(spec.allow) if spec.allow else None,
    }
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    try:
        snapshot_download(**kwargs)
    except HfHubHTTPError as exc:  # type: ignore
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status in {401, 403, 404}:
            raise RuntimeError(
                f"Failed fetching {spec.repo_id} (status {status}). "
                "Make sure the repo exists, you’re logged in (`huggingface-cli login`), "
                "and you’ve accepted the model’s license."
            ) from exc
        raise

    if not any(dest.iterdir()):
        raise RuntimeError(
            "Download produced no files. Make sure you're logged in and have accepted the model's license."
        )

    meta = {
        "key": spec.key,
        "repo_id": spec.repo_id,
        "notes": spec.notes,
    }
    (dest / "MODEL.json").write_text(json.dumps(meta, indent=2))
    print(f"[{spec.key}] installed at {dest}")


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        help="model keys to install (or 'all')",
        required=True,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="only print what would be done",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-download even if target exists",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list available models and exit",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)

    if args.list:
        list_models()
        return 0

    if "all" in args.models:
        target_keys = list(MODEL_REGISTRY.keys())
    else:
        target_keys = []
        for key in args.models:
            if key not in MODEL_REGISTRY:
                print(f"Unknown model key: {key}", file=sys.stderr)
                return 1
            target_keys.append(key)

    for key in target_keys:
        spec = MODEL_REGISTRY[key]
        try:
            install_model(spec, dry_run=args.dry_run, force=args.force)
        except Exception as exc:
            print(f"[{key}] ERROR: {exc}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
