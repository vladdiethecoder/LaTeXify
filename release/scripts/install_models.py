#!/usr/bin/env python3
"""Download and organize LaTeXify's local model dependencies."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

try:
    from huggingface_hub import snapshot_download  # type: ignore

    try:
        from huggingface_hub import HfHubHTTPError  # type: ignore
    except ImportError:
        from huggingface_hub.utils import HfHubHTTPError  # type: ignore
except Exception as exc:  # pragma: no cover
    print("[install_models] failed to import huggingface_hub:", exc, file=sys.stderr)
    snapshot_download = None  # type: ignore

    class HfHubHTTPError(Exception):  # type: ignore
        pass


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from release.core.model_paths import resolve_models_root  # noqa: E402

MODELS_DIR = resolve_models_root(REPO_ROOT / "models")


@dataclass(frozen=True)
class ModelSpec:
    key: str
    repo_id: Optional[str]
    target: Path
    allow: Optional[Iterable[str]] = None
    revision: Optional[str] = None
    manual_url: Optional[str] = None
    git_url: Optional[str] = None
    notes: Optional[str] = None

    def destination(self) -> Path:
        return MODELS_DIR / self.target


def _sanitize_model_subdir(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).lower()


def _repo_slug(repo_id: str) -> str:
    return repo_id.replace("/", "__")


INTERNVL_MODEL_ID = os.environ.get("LATEXIFY_INTERNVL_MODEL", "OpenGVLab/InternVL3_5-8B")
INTERNVL_TARGET = Path("ocr") / _sanitize_model_subdir(INTERNVL_MODEL_ID)
DEFAULT_LLM_REPO = os.environ.get("LATEXIFY_LLM_REPO", "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct")
LLM_DEFAULT_TARGET = Path("llm") / _repo_slug(DEFAULT_LLM_REPO)


MODEL_REGISTRY: Dict[str, ModelSpec] = {
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
        notes="llama.cpp-compatible judge (Q4_K_M). Requires accepting the Qwen 2.5 license.",
    ),
    "ocr/internvl": ModelSpec(
        key="ocr/internvl",
        repo_id=INTERNVL_MODEL_ID,
        target=INTERNVL_TARGET,
        notes=f"InternVL vision OCR (set LATEXIFY_INTERNVL_MODEL to override, default {INTERNVL_MODEL_ID}).",
    ),
    "ocr/florence-2-large": ModelSpec(
        key="ocr/florence-2-large",
        repo_id="microsoft/Florence-2-large-ft",
        target=Path("ocr/florence-2-large"),
        notes="Florence-2 Large fine-tuned for OCR/regional text (requires einops + timm).",
    ),
    "ocr/nougat-small": ModelSpec(
        key="ocr/nougat-small",
        repo_id="facebook/nougat-small",
        target=Path("ocr/nougat-small"),
        notes="Nougat LaTeX OCR specialist.",
    ),
    "ocr/trocr-math": ModelSpec(
        key="ocr/trocr-math",
        repo_id="microsoft/trocr-base-handwritten",
        target=Path("ocr/trocr-math"),
        notes="VisionEncoderDecoder math OCR (TrOCR).",
    ),
    "ocr/mineru-1.2b": ModelSpec(
        key="ocr/mineru-1.2b",
        repo_id=None,
        target=Path("ocr/mineru-1.2b"),
        git_url="https://github.com/opendatalab/MinerU.git",
        notes="Cloned from GitHub (opendatalab/MinerU). Contains OCR checkpoints.",
    ),
    "ocr/pix2tex-base": ModelSpec(
        key="ocr/pix2tex-base",
        repo_id="lupantech/pix2tex-base",
        target=Path("ocr/pix2tex-base"),
        notes="pix2tex LaTeX OCR recognizer (required for MathOCREngine).",
    ),
    "layout/layoutlmv3-base": ModelSpec(
        key="layout/layoutlmv3-base",
        repo_id="microsoft/layoutlmv3-base",
        target=Path("layout/layoutlmv3-base"),
        notes="LayoutLMv3-base checkpoint for DocumentStructureAnalyzer.",
    ),
    "llm/codellama-13b-math-gguf": ModelSpec(
        key="llm/codellama-13b-math-gguf",
        repo_id=None,
        target=Path("llm/codellama-13b-math-gguf"),
        manual_url="https://huggingface.co/TheBloke/CodeLlama-13B-Instruct-GGUF",
        notes=(
            "Download a codellama-13b-math GGUF file (e.g., Q4_K_M) and place it in this directory. "
            "Point --llama-cpp-model at the .gguf path."
        ),
    ),
    "llm/deepseek-coder-v2-lite": ModelSpec(
        key="llm/deepseek-coder-v2-lite",
        repo_id=DEFAULT_LLM_REPO,
        target=LLM_DEFAULT_TARGET,
        notes="Default DeepSeek refiner (7B). Requires `huggingface-cli login` and ~15 GB disk.",
    ),
}


def ensure_hf_available() -> None:
    if snapshot_download is None:
        raise RuntimeError("huggingface_hub is required. Install via `pip install huggingface_hub` inside your venv.")


def list_models() -> None:
    print("Available models:")
    for key, spec in MODEL_REGISTRY.items():
        print(f"- {key}")
        if spec.repo_id:
            print(f"  repo:   {spec.repo_id}")
        print(f"  target: {spec.destination()}")
        if spec.notes:
            print(f"  notes:  {spec.notes}")
        if spec.manual_url:
            print(f"  manual: {spec.manual_url}")
        print()


def install_model(spec: ModelSpec, *, dry_run: bool = False, force: bool = False) -> None:
    dest = spec.destination()
    if spec.git_url:
        if dry_run:
            print(f"[dry-run] {spec.key}: clone {spec.git_url} -> {dest}")
            return
        if dest.exists() and (not force) and any(dest.iterdir()):
            print(f"[{spec.key}] already exists at {dest} (use --force to re-download)")
            return
        if dest.exists():
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{spec.key}] cloning {spec.git_url} -> {dest}")
        try:
            subprocess.run(["git", "clone", "--depth", "1", spec.git_url, str(dest)], check=True)
        except FileNotFoundError as exc:  # pragma: no cover
            raise RuntimeError("git is required to clone repositories. Please install git.") from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"git clone failed for {spec.git_url}: {exc}") from exc
        git_dir = dest / ".git"
        if git_dir.exists():
            shutil.rmtree(git_dir)
        (dest / "MODEL.json").write_text(
            json.dumps({"key": spec.key, "git_url": spec.git_url, "notes": spec.notes}, indent=2),
            encoding="utf-8",
        )
        print(f"[{spec.key}] installed via git clone")
        return
    if spec.repo_id is None:
        dest.mkdir(parents=True, exist_ok=True)
        manual = dest / "MANUAL_DOWNLOAD.txt"
        if not manual.exists() or force:
            manual.write_text(
                f"This model must be downloaded manually.\nURL: {spec.manual_url or 'N/A'}\nNotes: {spec.notes or ''}\n",
                encoding="utf-8",
            )
        print(f"[{spec.key}] wrote manual instructions -> {manual}")
        return
    dest_has_payload = dest.exists() and any(dest.iterdir())
    if dry_run:
        verb = "skip (already present)" if dest_has_payload else f"download {spec.repo_id}"
        print(f"[dry-run] {spec.key}: {verb}")
        return
    ensure_hf_available()
    if dest_has_payload and not force:
        print(f"[{spec.key}] already exists at {dest} (use --force to re-download)")
        return
    if dest.exists() and force:
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[{spec.key}] downloading {spec.repo_id} -> {dest}")
    kwargs = {
        "repo_id": spec.repo_id,
        "local_dir": str(dest),
        "local_dir_use_symlinks": False,
        "resume_download": True,
        "allow_patterns": list(spec.allow) if spec.allow else None,
        "revision": spec.revision,
    }
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    try:
        snapshot_download(**kwargs)
    except HfHubHTTPError as exc:  # type: ignore
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status in {401, 403, 404}:
            raise RuntimeError(
                f"{spec.repo_id} denied (status {status}). Run `huggingface-cli login` and accept the repo license."
            ) from exc
        raise
    if not any(dest.iterdir()):
        raise RuntimeError("Download completed but directory is empty. Check your HF login and repo access.")
    (dest / "MODEL.json").write_text(
        json.dumps({"key": spec.key, "repo_id": spec.repo_id, "notes": spec.notes}, indent=2),
        encoding="utf-8",
    )
    print(f"[{spec.key}] installed at {dest}")


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True, help="Model keys to install (or 'all').")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without downloading.")
    parser.add_argument("--force", action="store_true", help="Re-download even if target exists.")
    parser.add_argument("--list", action="store_true", help="List all model keys and exit.")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    if args.list:
        list_models()
        return 0
    keys = MODEL_REGISTRY.keys() if "all" in args.models else args.models
    for key in keys:
        spec = MODEL_REGISTRY.get(key)
        if spec is None:
            print(f"Unknown model key: {key}", file=sys.stderr)
            return 1
        install_model(spec, dry_run=args.dry_run, force=args.force)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
