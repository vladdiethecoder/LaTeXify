#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DCL Synthesis Agent (budget-aware, optional reranking).

- Reads plan.json (tasks incl. Q-IDs).
- For each task, dynamically queries the appropriate indexes.
- Budget loop: initial topK per tool, optional expand rounds, final rerank to pick K.
- Deterministic LaTeX templates when --model none (default).
  Later you can pass a local HF model path and switch to greedy decoding.

CLI highlights:
  --init_topk, --expand_topk, --max_rounds, --final_k
  --reranker (e.g., BAAI/bge-reranker-v2-m3), --rerank_k
  --max_ctx_tokens (cap context pasted into template)

Outputs:
  build_dcl/snippets/<TASK>.tex

References:
- BGE reranker usage & model list (FlagEmbedding / HF).  # see docs
- Greedy decoding behavior in HF `generate` (no sampling).  # see docs
- ReAct / tool-use loop motivation.  # see docs
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

from dcl_tools import FaissSearcher, cap_context
from rerankers import get_reranker

logger = logging.getLogger(__name__)


def load_plan(path: str) -> Dict[str, Any]:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def _ensure_out_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _q_anchor(task: Dict[str, Any]) -> str:
    # Prefer an explicit title; fallback to the id
    title = task.get("title") or task.get("id", "")
    return str(title).strip()


def _gather_contexts(
    q: str,
    searchers: Dict[str, FaissSearcher],
    init_topk: int,
    expand_topk: int,
    max_rounds: int,
    reranker_name: str,
    rerank_k: int,
    final_k: int,
    max_ctx_tokens: int,
) -> Dict[str, List[str]]:
    """
    Budgeted DCL loop:
      Round 0: assessment + rubric
      Round 1..N: user + assignment expansions
      Optional rerank of pooled candidates, then cap by tokens.
    """
    pools: Dict[str, List[Dict[str, Any]]] = {k: [] for k in ["assessment", "rubric", "user", "assignment"]}

    # Round 0: must-know question + rubric
    pools["assessment"] = searchers["assessment"].search(q, top_k=init_topk)
    pools["rubric"] = searchers["rubric"].search(q, top_k=max(1, init_topk // 2))

    # Expansion rounds: pull from user/assignment
    for _ in range(max(0, max_rounds)):
        if expand_topk > 0:
            pools["user"].extend(searchers["user"].search(q, top_k=expand_topk))
            pools["assignment"].extend(searchers["assignment"].search(q, top_k=max(1, expand_topk // 2)))

    # Flatten all candidates with provenance
    candidates: List[Dict[str, Any]] = []
    for src, items in pools.items():
        for it in items:
            it2 = dict(it)
            it2["source"] = src
            candidates.append(it2)

    # Optional rerank
    rr = get_reranker(reranker_name)
    if candidates:
        scored = rr.rerank(q, candidates, top_k=rerank_k if rerank_k > 0 else len(candidates))
        # Keep per-source bins after rerank selection
        selected = scored[: final_k if final_k > 0 else len(scored)]
    else:
        selected = []

    # Prepare capped text contexts by source (keep order as in selection)
    by_src: Dict[str, List[str]] = {"assessment": [], "rubric": [], "user": [], "assignment": []}
    for s in selected:
        src = s.payload.get("source", "user")
        by_src[src].append(s.text)

    # Cap total context tokens per source (even split heuristic)
    if max_ctx_tokens > 0:
        per_bucket = max_ctx_tokens // 4
        for src in by_src:
            by_src[src] = cap_context(by_src[src], per_bucket)

    return by_src


def _emit_snippet(task_id: str, ctx: Dict[str, List[str]], out_dir: Path):
    """
    Deterministic LaTeX template that uses the contexts as evidence.
    Compatible with your preamble policy (no extra packages).
    """
    sec_name = task_id.replace("_", r"\_")
    lines: List[str] = []
    lines.append(f"\\section*{{{sec_name}}}")

    def _blk(title: str, items: List[str]):
        if not items:
            return
        lines.append(f"\\paragraph{{{title}}}")
        lines.append("\\begin{itemize}")
        for t in items:
            # sanitize minimal LaTeX specials
            safe = t.replace("%", "\\%")
            lines.append(f"  \\item {safe}")
        lines.append("\\end{itemize}")

    _blk("Question", ctx.get("assessment", []))
    _blk("Rubric", ctx.get("rubric", []))
    _blk("User Draft (evidence)", ctx.get("user", []))
    _blk("Assignment Rules", ctx.get("assignment", []))

    tex = "\n".join(lines) + "\n"
    (out_dir / f"{task_id}.tex").write_text(tex, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--user", required=True)
    ap.add_argument("--assessment", required=True)
    ap.add_argument("--rubric", required=True)
    ap.add_argument("--assignment", required=True)
    ap.add_argument("--latexkb", required=False, help="unused in v1.3 (reserved)")
    ap.add_argument("--out_dir", required=True)

    # budgets
    ap.add_argument("--init_topk", type=int, default=3)
    ap.add_argument("--expand_topk", type=int, default=2)
    ap.add_argument("--max_rounds", type=int, default=1)
    ap.add_argument("--final_k", type=int, default=4)
    ap.add_argument("--max_ctx_tokens", type=int, default=1200)

    # reranker
    ap.add_argument("--reranker", type=str, default="none", help="e.g., BAAI/bge-reranker-v2-m3 or 'none'")
    ap.add_argument("--rerank_k", type=int, default=8)

    # model hook (not used when 'none')
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--model", type=str, default="none")

    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="[synth-dcl] %(message)s")

    plan = load_plan(args.plan)
    tasks: List[Dict[str, Any]] = plan.get("tasks", [])

    # load searchers
    searchers = {
        "user": FaissSearcher(args.user),
        "assessment": FaissSearcher(args.assessment),
        "rubric": FaissSearcher(args.rubric),
        "assignment": FaissSearcher(args.assignment),
    }

    out = Path(args.out_dir)
    _ensure_out_dir(out)

    # Preamble & Title as in prior versions
    (out / "PREAMBLE.tex").write_text("% Preamble is managed elsewhere (lix_textbook).\\n", encoding="utf-8")
    (out / "TITLE.tex").write_text("\\maketitle\n", encoding="utf-8")

    for t in tasks:
        task_id = str(t.get("id", "")).strip()
        if task_id in ("PREAMBLE", "TITLE") or not task_id:
            continue

        q = _q_anchor(t)
        ctx = _gather_contexts(
            q=q,
            searchers=searchers,
            init_topk=args.init_topk,
            expand_topk=args.expand_topk,
            max_rounds=args.max_rounds,
            reranker_name=args.reranker,
            rerank_k=args.rerank_k,
            final_k=args.final_k,
            max_ctx_tokens=args.max_ctx_tokens,
        )
        _emit_snippet(task_id, ctx, out)

    logger.info("DCL synthesis complete -> %s", args.out_dir)


if __name__ == "__main__":
    main()
