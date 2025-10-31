#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-end driver:
1) plan
2) write static snippets
3) synthesize Q*.tex with local llama.cpp (GGUF)
4) aggregate to main.tex
5) latexmk in build/

DCL hygiene: export TEXINPUTS, build-only latexmk, stub LiX fallback if missing.
"""
from __future__ import annotations
import argparse, json, os, pathlib, subprocess, sys, textwrap

ROOT = pathlib.Path(__file__).resolve().parents[1]

def run(cmd, cwd=None, env=None):
    print(f"+ (cwd={cwd or ROOT}) {' '.join(cmd)}")
    p = subprocess.run(cmd, cwd=cwd or ROOT, env=env, text=True)
    if p.returncode != 0:
        sys.exit(p.returncode)

def ensure_stub_class(build_dir: pathlib.Path, doc_class: str):
    # Write a tiny fallback class that defers to article
    stubs = build_dir / "_stubs"
    stubs.mkdir(parents=True, exist_ok=True)
    cls = stubs / f"{doc_class}.cls"
    if not cls.exists():
        cls.write_text(textwrap.dedent(f"""
        %% Auto-generated LiX fallback stub
        \\NeedsTeXFormat{{LaTeX2e}}
        \\ProvidesClass{{{doc_class}}}[2025/10/30 LiX fallback stub]
        \\LoadClass{{article}}
        """).strip()+"\n", encoding="utf-8")

def build_texinputs_env(build_dir: pathlib.Path) -> str:
    # Search order: build/_stubs, kb/offline/classes, kb/offline/latex, kb/offline/course, LiX
    search = [
        build_dir / "_stubs",
        ROOT / "kb" / "offline" / "classes",
        ROOT / "kb" / "offline" / "latex",
        ROOT / "kb" / "offline" / "course",
        ROOT / "LiX",
    ]
    parts = [str(p) for p in search if p.exists()]
    return ":".join(parts) + ":"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="build")
    ap.add_argument("--questions", default="Q1a,Q1b")
    ap.add_argument("--doc-class", default="lix_textbook")
    ap.add_argument("--title", default="Untitled")
    ap.add_argument("--author", default="Student X")
    ap.add_argument("--course", default="COURSE 101")
    ap.add_argument("--date", default="\\today")
    ap.add_argument("--force-plan", action="store_true")
    ap.add_argument("--mode", default="auto")

    # LLM params
    ap.add_argument("--hf-cache", default=None)
    ap.add_argument("--gguf-model", default=None)
    ap.add_argument("--n-gpu-layers", type=int, default=-1)
    ap.add_argument("--tensor-split", default="auto")
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=12345)

    args = ap.parse_args()

    build_dir = ROOT / args.out
    build_dir.mkdir(exist_ok=True, parents=True)

    # 1) Planner → plan.json
    plan_path = build_dir / "plan.json"
    if args.force_plan or not plan_path.exists():
        plan = {
            "doc_class": args.doc_class,
            "title": args.title,
            "author": args.author,
            "course": args.course,
            "date": args.date,
            "questions": [q.strip() for q in args.questions.split(",") if q.strip()],
            "question_texts": {q.strip(): f"Answer the question {q.strip()} clearly with steps."
                               for q in args.questions.split(",")}
        }
        plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
        print(f"Wrote plan → {plan_path}")
    else:
        print(f"Using existing plan → {plan_path}")

    # 2) Static snippets (preamble/title) + seed main.tex
    run([
        sys.executable, "scripts/write_static_snippets.py",
        "--plan", str(plan_path),
        "--snippets", str(build_dir / "snippets"),
        "--main", str(build_dir / "main.tex")
    ])

    # 3) Synthesis with local model
    run([
        sys.executable, "scripts/synth_latex.py",
        "--plan", str(plan_path),
        "--outdir", str(build_dir / "snippets"),
        "--hf-cache", args.hf_cache or "",
        "--gguf-model", args.gguf_model or "",
        "--n-gpu-layers", str(args.n_gpu_layers),
        "--tensor-split", args.tensor_split,
        "--ctx", str(args.ctx),
        "--seed", str(args.seed),
        "--mode", args.mode
    ])

    # 4) Aggregate snippets into main.tex
    run([
        sys.executable, "scripts/aggregate_tex.py",
        "--plan", str(plan_path),
        "--snippets", str(build_dir / "snippets"),
        "--out", str(build_dir / "main.tex")
    ])

    # 5) Ensure stub class if real one isn't available
    ensure_stub_class(build_dir, args.doc_class)

    # DCL: TEXINPUTS search order (document it)
    print("[DCL] TEXINPUTS search order:")
    for p in build_texinputs_env(build_dir).split(":"):
        if p: print(f"   - {p}")
    tex_env = os.environ.copy()
    tex_env["TEXINPUTS"] = build_texinputs_env(build_dir)
    print(f"[DCL] TEXINPUTS (exported): {tex_env['TEXINPUTS']}")

    # latexmk inside build/
    run(["latexmk", "-cd", "-C", "main.tex"], cwd=build_dir, env=tex_env)
    run([
            "latexmk", "-cd", "-g",
            "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"
        ],
        cwd=build_dir, env=tex_env
    )
    print(f"Pipeline complete → {build_dir / 'main.pdf'}")

if __name__ == "__main__":
    main()
