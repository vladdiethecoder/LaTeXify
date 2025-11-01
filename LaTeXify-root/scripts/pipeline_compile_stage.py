#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pipeline_compile_stage.py

Convenience entrypoint to run:
  clean → compile → (auto-fix?) → finalize

Use this if you don’t want to edit existing orchestrators yet.
You can invoke from CI or your current `scripts/pipeline.py` at the "Compile" stage.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
BUILD = PROJECT_ROOT / "build"
RUNS_ROOT = PROJECT_ROOT / "dev" / "runs"
MAIN_TEX = BUILD / "main.tex"


def run(cmd: list[str]) -> int:
    return subprocess.call(cmd)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument("--auto-fix", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args(argv)

    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")

    # 1) Clean
    rc = run([sys.executable, str(HERE / "clean_build.py"), "--run-id", run_id])
    if rc != 0:
        return rc

    # (Your synthesis/aggregation steps would run here)
    # e.g., planner → retrieval → synthesis → aggregate producing build/main.tex and /build/snippets/*

    # 2) Compile (+ auto-fix)
    rc = run([
        sys.executable, str(HERE / "compile_loop.py"),
        "--main-tex", str(MAIN_TEX),
        "--build-dir", str(BUILD),
        "--runs-root", str(RUNS_ROOT),
        "--run-id", run_id,
        "--auto-fix", str(args.auto_fix),
        "--seed", str(args.seed),
    ])

    return rc


if __name__ == "__main__":
    sys.exit(main())
