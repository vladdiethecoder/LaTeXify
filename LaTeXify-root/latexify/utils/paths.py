from __future__ import annotations

"""Centralized helpers for resolving repository paths."""

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DOCS_ROOT = _REPO_ROOT / "docs"


def repo_root() -> Path:
    """Absolute path to the repository root."""

    return _REPO_ROOT


def docs_root() -> Path:
    """Return the documentation directory (ensuring it exists)."""

    _DOCS_ROOT.mkdir(parents=True, exist_ok=True)
    return _DOCS_ROOT


def tasks_root() -> Path:
    """Return the prompt/tasks directory, raising if missing."""

    root = docs_root() / "tasks"
    if not root.exists():  # pragma: no cover - developer configuration issue
        raise FileNotFoundError(f"tasks directory not found at {root}")
    return root
