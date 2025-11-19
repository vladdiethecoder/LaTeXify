#!/usr/bin/env python3
"""Download and organize LaTeXify's local model dependencies."""
from __future__ import annotations

import argparse
import importlib.util
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


def _load_model_paths_module():
    """Load core.model_paths without importing the full release package.

    This avoids triggering heavy side effects (vLLM, OCR adapters, etc.) during model
    installation while still reusing the canonical resolve_models_root helper.
    """

    path = REPO_ROOT / "release" / "core" / "model_paths.py"
    module_name = "release_core_model_paths_for_install"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load model_paths module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_model_paths = _load_model_paths_module()
resolve_models_root = _model_paths.resolve_models_root  # type: ignore[attr-defined]

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
    requires_auth: bool = False

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
KIMI_K2_REPO = os.environ.get("LATEXIFY_KIMI_K2_REPO", "unsloth/Kimi-K2-Instruct-0905-GGUF")
KIMI_K2_TARGET = Path("llm/kimi-k2-instruct-gguf")
ALLOWED_KIMI_VARIANTS = {"Q4_K_M", "Q3_K_M", "Q5_K_M"}
DEFAULT_KIMI_VARIANT = "Q4_K_M"


def _resolve_kimi_variant() -> str:
    candidate = os.environ.get("LATEXIFY_KIMI_K2_VARIANT", DEFAULT_KIMI_VARIANT).strip().upper()
    if candidate not in ALLOWED_KIMI_VARIANTS:
        print(
            f"[install_models] Unknown LATEXIFY_KIMI_K2_VARIANT '{candidate}'. "
            f"Falling back to {DEFAULT_KIMI_VARIANT}.",
            file=sys.stderr,
        )
        return DEFAULT_KIMI_VARIANT
    return candidate


KIMI_K2_VARIANT = _resolve_kimi_variant()


def _kimi_allow_patterns() -> tuple[str, ...]:
    override = os.environ.get("LATEXIFY_KIMI_K2_ALLOW_PATTERN")
    if override:
        return (override,)
    return (f"**/Kimi-K2-Instruct-0905-{KIMI_K2_VARIANT}-*.gguf",)


KIMI_K2_ALLOW = _kimi_allow_patterns()


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "layout/qwen2.5-vl-32b": ModelSpec(
        key="layout/qwen2.5-vl-32b",
        repo_id="Qwen/Qwen2.5-VL-32B-Instruct",
        target=Path("layout/qwen2.5-vl-32b"),
        notes="Vision-language layout model for PDF/page understanding.",
    ),
    "ocr/internvl": ModelSpec(
        key="ocr/internvl",
        repo_id=INTERNVL_MODEL_ID,
        target=INTERNVL_TARGET,
        notes=f"InternVL vision OCR (set LATEXIFY_INTERNVL_MODEL to override, default {INTERNVL_MODEL_ID}).",
        requires_auth=True,
    ),
    "ocr/florence-2-large": ModelSpec(
        key="ocr/florence-2-large",
        repo_id="microsoft/Florence-2-large-ft",
        target=Path("ocr/florence-2-large"),
        notes="Florence-2 Large fine-tuned for OCR/regional text (requires einops + timm).",
        requires_auth=True,
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
    "vision/flux-fill": ModelSpec(
        key="vision/flux-fill",
        repo_id="black-forest-labs/Flux.1-Fill-dev",
        target=Path("vision/flux-fill"),
        notes="Flux.1 Fill diffusion checkpoint for render-aware reconstruction (optional).",
    ),
    "layout/surya": ModelSpec(
        key="layout/surya",
        repo_id="SuryaResearch/surya_layout",
        target=Path("layout/surya"),
        notes="Surya layout detector weights for render-aware ingestion.",
    ),
    "llm/deepseek-coder-v2-lite": ModelSpec(
        key="llm/deepseek-coder-v2-lite",
        repo_id=DEFAULT_LLM_REPO,
        target=LLM_DEFAULT_TARGET,
        notes="Default DeepSeek refiner (7B). Requires `huggingface-cli login` and ~15 GB disk.",
        requires_auth=True,
    ),
    "llm/kimi-k2-instruct-gguf": ModelSpec(
        key="llm/kimi-k2-instruct-gguf",
        repo_id=KIMI_K2_REPO,
        target=KIMI_K2_TARGET,
        allow=KIMI_K2_ALLOW,
        notes=(
            "Kimi-K2 Instruct (GGUF). Requires llama-cpp-python with CUDA support. "
            "Use LATEXIFY_KIMI_K2_REPO, LATEXIFY_KIMI_K2_VARIANT, or LATEXIFY_KIMI_K2_ALLOW_PATTERN to override."
        ),
        requires_auth=True,
    ),
    "llm/mixtral-8x7b-instruct": ModelSpec(
        key="llm/mixtral-8x7b-instruct",
        repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        target=Path("llm/mixtral-8x7b-instruct"),
        notes="Mixtral 8x7B Instruct checkpoint (Transformers-compatible).",
        requires_auth=True,
    ),
    "llm/deepseek-v3": ModelSpec(
        key="llm/deepseek-v3",
        repo_id="deepseek-ai/DeepSeek-V3",
        target=Path("llm/deepseek-ai__DeepSeek-V3"),
        notes="DeepSeek V3 checkpoint for vLLM (large; ensure >60 GB free disk).",
        requires_auth=True,
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
        if spec.requires_auth:
            print("  auth:   huggingface-cli login required")
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
    if spec.key == "llm/kimi-k2-instruct-gguf":
        # Kimi-K2 repo stores quantizations under subdirectories like Q4_K_M/.
        # Flatten one .gguf file into the target directory so bootstrap_env can discover it.
        gguf_files = sorted(dest.rglob("*.gguf"))
        if not gguf_files:
            raise RuntimeError(
                "Kimi-K2 GGUF download did not contain any .gguf files. "
                "Check LATEXIFY_KIMI_K2_REPO / LATEXIFY_KIMI_K2_ALLOW_PATTERN and your HF access."
            )
        primary = next((p for p in gguf_files if KIMI_K2_VARIANT in p.name), gguf_files[0])
        if primary.parent != dest:
            target = dest / primary.name
            if not target.exists():
                primary.rename(target)
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
