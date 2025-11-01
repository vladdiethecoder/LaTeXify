#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
clean_build.py

Project-level clean step to guarantee reproducibility by removing stale build artifacts.

Removes:
- /build/snippets
- /build/aux
- /build/logs
- /build/cache
- Common latex aux files under /build: *.aux, *.log, *.out, *.bbl, *.blg, *.toc, *.lof, *.lot, *.fls, *.fdb_latexmk

Also supports per-run id cleanup under /dev/runs/<run_id>.

Safety:
- Restricts deletions to within the repository root.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
BUILD = PROJECT_ROOT / "build"
RUNS_ROOT = PROJECT_ROOT / "dev" / "runs"


def safe_rm(path: Path) -> None:
    if not path.exists():
        return
    # Safety: ensure we are inside PROJECT_ROOT
    try:
        path = path.resolve()
        if PROJECT_ROOT not in path.parents and path != PROJECT_ROOT:
            print(f"Refusing to delete outside project: {path}", file=sys.stderr)
            return
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)
    except Exception as e:
        print(f"[clean] Failed to remove {path}: {e}", file=sys.stderr)


def clean_build(build_dir: Path = BUILD) -> int:
    # folders
    for d in ("snippets", "aux", "logs", "cache"):
        safe_rm(build_dir / d)

    # aux files (top-level)
    exts = [
        ".aux", ".log", ".out", ".bbl", ".blg", ".toc",
        ".lof", ".lot", ".fls", ".fdb_latexmk", ".synctex.gz"
    ]
    for p in build_dir.glob("*"):
        if p.suffix in exts:
            safe_rm(p)

    return 0


def clean_run(run_id: str | None) -> int:
    if not run_id:
        return 0
    safe_rm(RUNS_ROOT / run_id)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Clean build artifacts and (optionally) a specific run id")
    ap.add_argument("--build-dir", type=Path, default=BUILD)
    ap.add_argument("--run-id", type=str, default=None)
    args = ap.parse_args(argv)

    rc1 = clean_build(args.build_dir)
    rc2 = clean_run(args.run_id)
    return rc1 or rc2


if __name__ == "__main__":
    sys.exit(main())
