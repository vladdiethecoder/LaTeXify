from __future__ import annotations
import argparse, json, os, subprocess
from pathlib import Path
from latexify.kb.query_index import build_context_bundle
from latexify.pipeline.synth_latex import synthesize_snippet
from latexify.kb.ensure_kb_alias import ensure_kb_alias

def parse_indices(mapping: str) -> dict:
    out = {}
    for part in mapping.split():
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_id", required=True)
    ap.add_argument("--question", default="")
    ap.add_argument("--indices", required=True, help='e.g. "rubric=kb/latex assignment=kb/latex assessment=kb/latex user=kb/latex"')
    ap.add_argument("--plan", default="plan.json")
    ap.add_argument("--snippets_dir", default="snippets")
    ap.add_argument("--bundles_dir", default="bundles")
    ap.add_argument("--out_dir", default="build")
    ap.add_argument("--k_user", type=int, default=6)
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.snippets_dir, exist_ok=True)
    os.makedirs(args.bundles_dir, exist_ok=True)

    # Ensure canonical KB alias exists (uses offline/online sources if needed)
    ensure_kb_alias(Path("kb/latex"), [Path("kb/latex"), Path("kb/offline/latex"), Path("kb/online/latex")])

    indices = par
