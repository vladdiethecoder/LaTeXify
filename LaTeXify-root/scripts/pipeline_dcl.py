#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-end DCL pipeline (planner -> synth_dcl -> checks -> aggregate).

Usage (example):
  python scripts/pipeline_dcl.py \
    --assignment indexes/assignment.index \
    --assessment indexes/assessment.index \
    --rubric indexes/rubric.index \
    --user indexes/user.index \
    --latexkb indexes/latex_docs.index \
    --out build_dcl/ \
    --doc_class lix_textbook \
    --title " " --author " " --course " " --date " " \
    --qid_min 1 --qid_max 400 --fallback_user_ids \
    --init_topk 3 --expand_topk 2 --max_rounds 1 --final_k 4 \
    --reranker none --rerank_k 8 --max_ctx_tokens 1200 \
    --model none \
    --compile
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_cmd(args):
    print(f"[dcl-pipeline] $ {' '.join(args)}")
    subprocess.run(args, check=True)


def main():
    ap = argparse.ArgumentParser()
    # indexes
    ap.add_argument("--assignment", required=True)
    ap.add_argument("--assessment", required=True)
    ap.add_argument("--rubric", required=True)
    ap.add_argument("--user", required=True)
    ap.add_argument("--latexkb", required=True)

    # plan metadata
    ap.add_argument("--out", required=True)
    ap.add_argument("--doc_class", default="lix_textbook")
    ap.add_argument("--title", default="")
    ap.add_argument("--author", default="")
    ap.add_argument("--course", default="")
    ap.add_argument("--date", default="")

    # planner knobs
    ap.add_argument("--qid_min", type=int, default=None)
    ap.add_argument("--qid_max", type=int, default=None)
    ap.add_argument("--fallback_user_ids", action="store_true")

    # DCL budgets + reranker
    ap.add_argument("--init_topk", type=int, default=3)
    ap.add_argument("--expand_topk", type=int, default=2)
    ap.add_argument("--max_rounds", type=int, default=1)
    ap.add_argument("--final_k", type=int, default=4)
    ap.add_argument("--max_ctx_tokens", type=int, default=1200)
    ap.add_argument("--reranker", type=str, default="none")
    ap.add_argument("--rerank_k", type=int, default=8)

    # model hook
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--model", type=str, default="none")

    # compile
    ap.add_argument("--compile", action="store_true")

    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    plan_path = out_dir / "plan.json"
    snippets_dir = out_dir / "snippets"
    checks_dir = out_dir / "checks"
    checks_dir.mkdir(parents=True, exist_ok=True)

    # 1) planner
    planner_cmd = [
        "python", "scripts/planner_scaffold.py",
        "--assessment", args.assessment,
        "--rubric", args.rubric,
        "--assignment", args.assignment,
        "--user", args.user,
        "--out", str(plan_path),
        "--doc_class", args.doc_class,
        "--title", args.title,
        "--author", args.author,
        "--course", args.course,
        "--date", args.date,
    ]
    if args.qid_min is not None:
        planner_cmd += ["--qid_min", str(args.qid_min)]
    if args.qid_max is not None:
        planner_cmd += ["--qid_max", str(args.qid_max)]
    if args.fallback_user_ids:
        planner_cmd += ["--fallback_user_ids"]
    run_cmd(planner_cmd)

    # 2) synth (DCL)
    synth_cmd = [
        "python", "scripts/synth_latex_dcl.py",
        "--plan", str(plan_path),
        "--user", args.user,
        "--assessment", args.assessment,
        "--rubric", args.rubric,
        "--assignment", args.assignment,
        "--latexkb", args.latexkb,
        "--out_dir", str(snippets_dir),
        "--device", args.device or "",
        "--model", args.model or "none",
        "--init_topk", str(args.init_topk),
        "--expand_topk", str(args.expand_topk),
        "--max_rounds", str(args.max_rounds),
        "--final_k", str(args.final_k),
        "--max_ctx_tokens", str(args.max_ctx_tokens),
        "--reranker", args.reranker,
        "--rerank_k", str(args.rerank_k),
    ]
    run_cmd(synth_cmd)

    # 3) postflight checks (inject/report)
    checks_cmd = [
        "python", "scripts/postflight_tex_checks.py",
        "--snippets", str(snippets_dir),
        "--out", str(checks_dir / "report.json"),
        "--plan", str(plan_path),
    ]
    run_cmd(checks_cmd)

    # 4) aggregate + compile
    agg_cmd = [
        "python", "scripts/aggregate_tex.py",
        "--plan", str(plan_path),
        "--snippets", str(snippets_dir),
        "--out_dir", str(out_dir),
    ]
    if args.compile:
        agg_cmd.append("--compile")
    run_cmd(agg_cmd)

    print("[dcl-pipeline] Done.")


if __name__ == "__main__":
    main()
