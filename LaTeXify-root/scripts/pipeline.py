#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pipeline.py — Orchestrator (compile stage wired to Final Compilation Loop)

This keeps your existing upstream stages (Planner → Retrieval → Synthesis → Aggregation)
as-is, and replaces the raw latexmk call with:
  1) scripts/clean_build.py
  2) scripts/compile_loop.py  (auto-fix once)

If your upstream agents are invoked here, leave them; if they run elsewhere,
this file still works as a post-Aggregation compile driver when build/main.tex exists.

Usage:
  python pipeline.py --seed 4242
"""

from __future__ import annotations
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[1]          # << was likely HERE.parent before
BUILD = REPO_ROOT / "build"
RUNS  = REPO_ROOT / "dev" / "runs"

def run(cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> None:
    print(f"[run] {' '.join(map(str, cmd))}")
    subprocess.check_call(cmd, cwd=str(cwd or ROOT), env=env)

def ensure_dirs() -> None:
    BUILD.mkdir(parents=True, exist_ok=True)
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)

def make_tex_env() -> dict:
    env = os.environ.copy()
    # Ensure LiX classes and class KB (if you use TEXINPUTS)
    # Append . to TEXINPUTS so LaTeX can find local files first
    env["TEXINPUTS"] = env.get("TEXINPUTS", "") + (":" if env.get("TEXINPUTS") else "") + str(ROOT) + ":."
    return env

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument("--skip-agents", action="store_true",
                    help="Skip Planner/RAG/Synth/Aggregate and just compile existing build/main.tex")
    args = ap.parse_args()

    ensure_dirs()
    tex_env = make_tex_env()
    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")

    # ----------------------------------------------------------------------
    # (Optional) Upstream agents if you invoke them from here.
    # Commented as placeholders; leave your existing calls if you already do this elsewhere.
    if not args.skip_agents:
        # Example placeholders — uncomment/adapt if your CLI matches:
        # run([sys.executable, str(SCRIPTS / "planner_scaffold.py"), "--out", str(BUILD / "plan.json")], env=tex_env)
        # run([sys.executable, str(SCRIPTS / "retrieval_agent.py"), "--plan", str(BUILD / "plan.json"), "--out", "bundles"], env=tex_env)
        # run([sys.executable, str(SCRIPTS / "synth_latex.py"), "--bundles", "bundles", "--out", str(BUILD / "snippets")], env=tex_env)
        # run([sys.executable, str(SCRIPTS / "aggregator.py"), "--plan", str(BUILD / "plan.json"), "--snippets", str(BUILD / "snippets"), "--out", str(BUILD)], env=tex_env)
        pass

    # Require a main.tex to compile
    main_tex = BUILD / "main.tex"
    if not main_tex.exists():
        print(f"[pipeline] Expected {main_tex} but it does not exist. Did Aggregation run?")
        return 1

    # ----------------------------------------------------------------------
    # NEW: Clean → Compile → (Auto-Fix once)
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

    print(f"[pipeline] Done → {BUILD / 'main.pdf'}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
