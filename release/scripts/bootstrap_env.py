#!/usr/bin/env python3
"""One-click bootstrapper for the LaTeXify release environment."""
from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional

RELEASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = RELEASE_DIR.parent
DEFAULT_VENV = RELEASE_DIR / ".venv"
REQUIREMENTS_FILE = RELEASE_DIR / "requirements.txt"
KIMI_MODEL_KEY = "llm/kimi-k2-instruct-gguf"
ALLOWED_KIMI_VARIANTS = {"Q4_K_M", "Q3_K_M", "Q5_K_M"}
DEFAULT_KIMI_VARIANT = "Q4_K_M"
REQUIRED_MODELS = [
    "layout/qwen2.5-vl-32b",
    "ocr/internvl",
    "ocr/florence-2-large",
    "ocr/nougat-small",
    "ocr/mineru-1.2b",
    "llm/mixtral-8x7b-instruct",
    "llm/deepseek-v3",
    KIMI_MODEL_KEY,
]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from release.core.model_paths import resolve_models_root  # noqa: E402

MODELS_ROOT = resolve_models_root(REPO_ROOT / "models")
KIMI_MODEL_SUBDIR = Path("llm") / "kimi-k2-instruct-gguf"


def _resolve_kimi_variant() -> str:
    candidate = os.environ.get("LATEXIFY_KIMI_K2_VARIANT", DEFAULT_KIMI_VARIANT).strip().upper()
    if candidate not in ALLOWED_KIMI_VARIANTS:
        print(
            f"[bootstrap] Unknown Kimi variant '{candidate}'. Falling back to {DEFAULT_KIMI_VARIANT}.",
            file=sys.stderr,
        )
        return DEFAULT_KIMI_VARIANT
    return candidate


KIMI_VARIANT = _resolve_kimi_variant()
KIMI_PATTERN_HINT = os.environ.get("LATEXIFY_KIMI_K2_ALLOW_PATTERN") or f"Kimi-K2-Instruct-0905-{KIMI_VARIANT}"
KIMI_MODEL_FILENAME = os.environ.get("LATEXIFY_KIMI_K2_FILENAME", f"Kimi-K2-Instruct-0905-{KIMI_VARIANT}.gguf")


def _venv_bin(venv_path: Path, executable: str) -> Path:
    if os.name == "nt":
        return venv_path / "Scripts" / (executable + ("" if executable.endswith(".exe") else ".exe"))
    return venv_path / "bin" / executable


def _run(cmd: List[str], *, env: dict | None = None) -> None:
    print("[bootstrap]", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def _ensure_venv(venv_path: Path, python_exe: str) -> tuple[Path, Path]:
    if not venv_path.exists():
        print(f"[bootstrap] creating venv at {venv_path}")
        _run([python_exe, "-m", "venv", str(venv_path)])
    pip_bin = _venv_bin(venv_path, "pip")
    python_bin = _venv_bin(venv_path, "python")
    if not pip_bin.exists() or not python_bin.exists():
        raise RuntimeError(f"Broken virtualenv at {venv_path}; delete it and rerun.")
    return pip_bin, python_bin


def _install_python_deps(pip_bin: Path, extras: Iterable[str], upgrade_pip: bool) -> None:
    if upgrade_pip:
        _run([str(pip_bin), "install", "--upgrade", "pip"])
    _run([str(pip_bin), "install", "-r", str(REQUIREMENTS_FILE)])
    if extras:
        _run([str(pip_bin), "install", *extras])


def _locate_installer() -> Path:
    candidates = [
        REPO_ROOT / "release" / "scripts" / "install_models.py",
        REPO_ROOT / "scripts" / "install_models.py",
        REPO_ROOT / "LaTeXify-root" / "scripts" / "install_models.py",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not locate install_models.py. Expected it under 'release/scripts/' or a top-level 'scripts/' directory."
    )


def _load_installer_module():
    installer_path = _locate_installer()
    module_name = "release_install_models"
    spec = importlib.util.spec_from_file_location(module_name, installer_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load installer module from {installer_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _install_models(models: Iterable[str], force: bool, dry_run: bool) -> None:
    install_models = _load_installer_module()
    registry = install_models.MODEL_REGISTRY
    for key in models:
        spec = registry.get(key)
        if spec is None:
            print(f"[bootstrap] skipping unknown model key '{key}'")
            continue
        install_models.install_model(spec, dry_run=dry_run, force=force)


def _discover_kimi_model() -> Optional[Path]:
    override = os.environ.get("LATEXIFY_KIMI_K2_MODEL")
    if override:
        candidate = Path(override).expanduser()
        if candidate.exists():
            return candidate
    search_roots = []
    env_dir = os.environ.get("LATEXIFY_KIMI_K2_MODEL_DIR")
    if env_dir:
        search_roots.append(Path(env_dir).expanduser())
    search_roots.append(MODELS_ROOT / KIMI_MODEL_SUBDIR)
    for base in search_roots:
        if base is None:
            continue
        if base.is_file() and base.suffix == ".gguf":
            return base
        if not base.exists() or not base.is_dir():
            continue
        candidates = sorted(base.glob("*.gguf"))
        if not candidates:
            continue
        preferred = KIMI_PATTERN_HINT
        prioritized = [cand for cand in candidates if preferred and preferred in cand.name]
        if prioritized:
            return prioritized[0]
        return candidates[0]
    return None


def _verify_kimi_model() -> None:
    model_path = _discover_kimi_model()
    if model_path is None:
        print(
            "[bootstrap] Kimi-K2 GGUF missing; run with --models llm/kimi-k2-instruct-gguf or set LATEXIFY_KIMI_K2_MODEL.",
        )
        return
    try:
        from release.models.kimi_k2_adapter import GGUFModelConfig, KimiK2InstructAdapter
    except ImportError as exc:  # pragma: no cover - bootstrap fallback
        print(f"[bootstrap] skipping Kimi-K2 verification (llama-cpp unavailable): {exc}")
        return
    try:
        adapter = KimiK2InstructAdapter(GGUFModelConfig(model_path=model_path))
        adapter.warmup()
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize Kimi-K2 GGUF at {model_path}: {exc}") from exc
    print(f"[bootstrap] verified llama-cpp loading for {model_path}")


def _verify_vllm_client() -> None:
    if os.environ.get("LATEXIFY_DISABLE_VLLM", "0") == "1":
        print("[bootstrap] skipping vLLM verification (LATEXIFY_DISABLE_VLLM=1).")
        return
    try:
        from release.models.vllm_client import get_vllm_client
    except ImportError as exc:  # pragma: no cover - optional dependency
        print(f"[bootstrap] skipping vLLM verification (vllm unavailable: {exc})")
        return
    client = get_vllm_client()
    if client is None:
        raise RuntimeError("vLLM client unavailable; set LATEXIFY_VLLM_MODEL or install vllm to verify.")
    try:
        reply = client.generate("Respond READY", max_tokens=4, temperature=0.0).strip()
    except Exception as exc:  # pragma: no cover - GPU/driver failures
        raise RuntimeError(f"vLLM verification failed: {exc}") from exc
    print(f"[bootstrap] verified vLLM client ({reply or 'no text'})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--venv",
        type=Path,
        default=DEFAULT_VENV,
        help="Path to the virtualenv to create/use (default: release/.venv).",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used to create the virtualenv.",
    )
    parser.add_argument(
        "--extras",
        nargs="*",
        default=[],
        help="Additional pip packages to install after requirements.txt.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=REQUIRED_MODELS,
        help="Model keys to install via scripts/install_models.py.",
    )
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="Skip model downloads (only install Python dependencies).",
    )
    parser.add_argument(
        "--force-models",
        action="store_true",
        help="Force re-download of models even if directories already exist.",
    )
    parser.add_argument(
        "--dry-run-models",
        action="store_true",
        help="Show which models would download without fetching payloads.",
    )
    parser.add_argument(
        "--verify-vllm",
        action="store_true",
        help="Attempt to start a vLLM client after installation to ensure the configured model loads.",
    )
    parser.add_argument(
        "--no-upgrade-pip",
        dest="upgrade_pip",
        action="store_false",
        help="Skip the initial `pip install --upgrade pip` step.",
    )
    parser.set_defaults(upgrade_pip=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pip_bin, py_bin = _ensure_venv(args.venv, args.python)
    _install_python_deps(pip_bin, args.extras, args.upgrade_pip)
    if not args.skip_models:
        _install_models(args.models, force=args.force_models, dry_run=args.dry_run_models)
    _verify_kimi_model()
    if args.verify_vllm:
        _verify_vllm_client()
    print("[bootstrap] complete.")
    print(f"[bootstrap] activate via: source {args.venv}/bin/activate" if os.name != "nt" else f"[bootstrap] activate via: {args.venv}\\Scripts\\activate.bat")
    print(f"[bootstrap] python: {py_bin}")


if __name__ == "__main__":
    main()
