#!/usr/bin/env python3
"""One-click bootstrapper for the LaTeXify release environment."""
from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List

RELEASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = RELEASE_DIR.parent
DEFAULT_VENV = RELEASE_DIR / ".venv"
REQUIREMENTS_FILE = RELEASE_DIR / "requirements.txt"
REQUIRED_MODELS = [
    "layout/qwen2.5-vl-32b",
    "judge/qwen2.5-72b-gguf",
    "ocr/internvl",
    "ocr/florence-2-large",
    "ocr/nougat-small",
    "ocr/mineru-1.2b",
]


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
    print("[bootstrap] complete.")
    print(f"[bootstrap] activate via: source {args.venv}/bin/activate" if os.name != "nt" else f"[bootstrap] activate via: {args.venv}\\Scripts\\activate.bat")
    print(f"[bootstrap] python: {py_bin}")


if __name__ == "__main__":
    main()
