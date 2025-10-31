#!/usr/bin/env python3
"""
Remove historical LaTeX byproducts that landed in repo root (or other dirs)
from earlier experiments. Current pipeline already confines outputs to build/.
Run once from repo root.

Usage:
  python scripts/clean_strays.py
"""
from __future__ import annotations
from pathlib import Path
import fnmatch
import os

# Only clean outside 'build' to avoid touching current outputs
PATTERNS = [
    "*.aux", "*.fls", "*.fdb_latexmk", "*.log", "*.out",
    "*.synctex*", "*.toc", "*.bbl", "*.blg",
]

EXCLUDE_DIRS = {"build", ".venv", ".venv-faiss311", ".git", "__pycache__"}

def should_skip_dir(p: Path) -> bool:
    name = p.name
    if name in EXCLUDE_DIRS:
        return True
    # Skip any directory inside 'build'
    if "build" in p.parts:
        return True
    return False

def rm_strays(root: Path) -> list[Path]:
    removed: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        d = Path(dirpath)
        # mutate dirnames to prune traversal
        dirnames[:] = [n for n in dirnames if not should_skip_dir(d / n)]
        for fname in filenames:
            f = d / fname
            if any(fnmatch.fnmatch(fname, pat) for pat in PATTERNS):
                try:
                    f.unlink()
                    removed.append(f)
                except Exception:
                    pass
    return removed

if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    removed = rm_strays(repo_root)
    if removed:
        print("[clean-strays] Removed:")
        for p in removed:
            print(f"  - {p}")
    else:
        print("[clean-strays] Nothing to remove.")
