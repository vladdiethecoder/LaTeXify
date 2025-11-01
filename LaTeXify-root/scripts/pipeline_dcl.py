#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pipeline_dcl.py — Orchestrator variant (Document Class / DCL flow)

This variant is identical in spirit to pipeline.py but is a separate entrypoint if you
have a DCL-specific builder. It calls the Final Compilation Loop to compile build/main.tex.
"""

from __future__ import annotations
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
BUILD = ROOT / "build"
RUNS_ROOT = ROOT / "dev" / "runs"

def run(cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> None:
    print(f"[run] {' '.join(map(str, cmd))}")
    subprocess.check_call(cmd, cwd=str(cwd or ROOT), env=env)

def ensure_dirs() -> None:
    BUILD.mkdir(parents=True, exist_ok=True)
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)

def make_tex_env() -> dict:
    env = os.environ.copy()
    env["TEXINPUTS"] = env.get("TEXINPUTS", "") + (":" if env.get("TEXINPUTS") else "") + str(ROOT) + ":."
    return env

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument("--skip-aggregate", action="store_true",
                    help="Skip Aggregation; just compile existing build/main.tex")
    args = ap.parse_args()

    ensure_dirs()
    tex_env = make_tex_env()
    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")

    if not args.skip_aggregate:
        # Example: if DCL pipeline produces build/main.tex here; adapt if needed.
        # run([sys.executable, str(SCRIPTS / "aggregator.py"), "--plan", str(BUILD / "plan.json"), "--snippets", str(BUILD / "snippets"), "--out", str(BUILD)], env=tex_env)
        pass

    main_tex = BUILD / "main.tex"
    if not main_tex.exists():
        print(f"[pipeline_dcl] Expected {main_tex} but it does not exist. Did Aggregation run?")
        return 1

    # Clean → Compile → (Auto-Fix once)
    run([sys.executable, str(SCRIPTS / "clean_build.py")], env=tex_env)
    run([
        sys.executable, str(SCRIPTS / "compile_loop.py"),
        "--main-tex", str(main_tex),
        "--build-dir", str(BUILD),
        "--runs-root", str(RUNS_ROOT),
        "--run-id", run_id,
        "--auto-fix", "1",
        "--seed", str(args.seed),
    ], env=tex_env)

    print(f"[pipeline_dcl] Done → {BUILD / 'main.pdf'}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
