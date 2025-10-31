#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rubric_cover.py (v1.2)

Compute a simple rubric-coverage report by checking whether each rubric
criterion's key terms appear in the synthesized snippet for each task.

Inputs:
  --plan build/plan.json
  --snippets build/snippets
  --context build/context        # produced by retrieval_agent.py
Outputs:
  build/reports/rubric_coverage.json

This is heuristic but useful to spot uncovered rubric rows quickly.
"""
from __future__ import annotations

import re
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any


def _task_ids(plan_path: Path) -> List[str]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    return [t["id"] for t in plan.get("tasks", []) if t.get("type") == "question"]


def _load_context(ctx_dir: Path, tid: str) -> Dict[str, Any]:
    p = ctx_dir / f"{tid}.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _key_terms(text: str) -> List[str]:
    # Extract crude key terms: lowercase words >= 5 chars (drop LaTeX)
    t = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', ' ', text)  # strip \cmd{...}
    t = re.sub(r'[^a-zA-Z0-9\s]', ' ', t).lower()
    words = [w for w in t.split() if len(w) >= 5]
    # keep top-N unique in order
    seen = set()
    out = []
    for w in words:
        if w not in seen:
            seen.add(w)
            out.append(w)
        if len(out) >= 12:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--snippets", required=True)
    ap.add_argument("--context", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sdir = Path(args.snippets)
    cdir = Path(args.context)

    report: Dict[str, Any] = {"tasks": {}}

    for tid in _task_ids(Path(args.plan)):
        snip_path = sdir / f"{tid}.tex"
        ctx = _load_context(cdir, tid)
        rub = ctx.get("rubric", {}) or {}
        criteria = rub.get("criteria", [])

        snippet = ""
        if snip_path.exists():
            snippet = snip_path.read_text(encoding="utf-8").lower()

        coverage = []
        for row in criteria:
            verb = (row.get("verbatim") or "").lower()
            if not verb:
                continue
            terms = _key_terms(verb)
            hits = [t for t in terms if t in snippet]
            coverage.append({
                "criterion_id": row.get("id"),
                "label": row.get("label"),
                "terms": terms,
                "hits": hits,
                "hit_ratio": (len(hits) / max(len(terms), 1)),
            })

        report["tasks"][tid] = {
            "criteria_count": len(criteria),
            "covered": sum(1 for c in coverage if c["hit_ratio"] >= 0.4),
            "details": coverage,
        }

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[rubric] wrote {args.out}")

if __name__ == "__main__":
    main()
