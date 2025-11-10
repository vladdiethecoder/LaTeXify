from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

_SUPPORTED_MINOR_MIN = 10
_SUPPORTED_MINOR_MAX = 14  # exclusive
_PYTHON_TOKEN = "LATEXIFY_BOOTSTRAP_PY"
_VENV_TOKEN = "LATEXIFY_BOOTSTRAP_VENV"


def _version_ok(version: str) -> bool:
    parts = version.split(".")
    if len(parts) < 2:
        return False
    try:
        major, minor = int(parts[0]), int(parts[1])
    except ValueError:
        return False
    return major == 3 and _SUPPORTED_MINOR_MIN <= minor < _SUPPORTED_MINOR_MAX


def _resolve_python(candidate: str | None) -> str | None:
    if not candidate:
        return None
    path = shutil.which(candidate)
    if path:
        return path
    target = Path(candidate)
    if target.exists():
        return str(target.resolve())
    return None


def ensure_supported_python(script_path: Path, argv: Sequence[str], preferred_python: str | None = None) -> None:
    if os.environ.get(_PYTHON_TOKEN) == "1":
        return
    if _version_ok(sys.version.split()[0]):
        return
    env_pref = preferred_python or os.environ.get("LATEXIFY_PYTHON_BIN")
    candidates: list[str] = []
    resolved = _resolve_python(env_pref)
    if resolved:
        candidates.append(resolved)
    for minor in (13, 12, 11, 10):
        resolved_candidate = _resolve_python(f"python3.{minor}")
        if resolved_candidate and resolved_candidate not in candidates:
            candidates.append(resolved_candidate)
    for candidate in candidates:
        try:
            version = subprocess.check_output(
                [candidate, "-c", "import sys; print(sys.version.split()[0])"],
                text=True,
            ).strip()
        except Exception:
            continue
        if not _version_ok(version):
            continue
        env = os.environ.copy()
        env[_PYTHON_TOKEN] = "1"
        cmd = [candidate, str(script_path), *argv[1:]]
        print(f"[env] Re-executing under {candidate} (Python {version})")
        os.execve(cmd[0], cmd, env)
    raise RuntimeError(
        "LaTeXify requires Python 3.10–3.13 to run GPU backends. "
        "Install python3.12 and either run this script with it or pass --python-bin /path/to/python3.12."
    )


def _in_virtualenv() -> bool:
    base_prefix = getattr(sys, "base_prefix", sys.prefix)
    return sys.prefix != base_prefix


def ensure_virtualenv(venv_path: Path, script_path: Path, argv: Sequence[str]) -> None:
    if os.environ.get(_VENV_TOKEN) == "1" or _in_virtualenv():
        return
    venv_path = venv_path.expanduser()
    if not venv_path.exists():
        print(f"[env] Creating virtualenv at {venv_path}")
        subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)
    python_bin = venv_path / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt" else "python")
    if not python_bin.exists():
        raise FileNotFoundError(f"Could not locate python executable in {venv_path}")
    env = os.environ.copy()
    env[_VENV_TOKEN] = "1"
    cmd = [str(python_bin), str(script_path), *argv[1:]]
    print(f"[env] Re-executing inside virtualenv {venv_path}")
    os.execve(cmd[0], cmd, env)


__all__ = ["ensure_supported_python", "ensure_virtualenv"]
