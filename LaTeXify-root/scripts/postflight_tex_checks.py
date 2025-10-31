#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
postflight_tex_checks.py (v1.1.1)

Static lint for LaTeX snippets. Emits JSON report and optionally injects \todo{} lines.

Upgrades vs v1.1:
- Accepts --plan and restricts scanning to the snippet IDs enumerated in plan.json.
- Avoids reporting on stale .tex files left from previous runs.

Checks:
- Equation alignment: multi-line equations with '=' outside align/aligned.
- Units: numbers with common units not wrapped in \SI{}{} or \si{}.
- Labels: figure/table/equation environments missing \label{...}.

Usage:
  python scripts/postflight_tex_checks.py \
    --snippets build/snippets \
    --out build/checks/report.json \
    --apply_todos \
    [--plan build/plan.json]
"""
from __future__ import annotations

import re
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any

UNIT_RX = re.compile(
    r"(?<!\\SI\{|\\si\{)"
    r"(\\?)"
    r"(\d+(?:\.\d+)?)\s*"
    r"(m|cm|mm|km|s|ms|kg|g|N|J|W|Pa|Hz|m/s|m\\,/s|\\mathrm\{[A-Za-z]+\})"
)
EQN_ENV_RX = re.compile(r"\\begin\{(equation\*?|gather\*?|multline\*?)\}(.*?)\\end\{\1\}", re.S)
ALIGN_ENV_RX = re.compile(r"\\begin\{(align\*?|aligned)\}")
LABEL_RX = re.compile(r"\\label\{[^}]+\}")

def detect_units(snippet: str) -> List[str]:
    hits = []
    for m in UNIT_RX.finditer(snippet):
        hits.append(m.group(0))
    return hits

def needs_align(snippet: str) -> bool:
    if ALIGN_ENV_RX.search(snippet):
        return False
    for m in EQN_ENV_RX.finditer(snippet):
        body = m.group(2)
        if body.count("=") >= 2 and "\n" in body:
            return True
    return False

def missing_labels(snippet: str) -> List[str]:
    out = []
    for env in ("figure", "table", "equation"):
        pattern = re.compile(rf"\\begin\{{{env}\}}(.*?)\\end\{{{env}\}}", re.S)
        for m in pattern.finditer(snippet):
            if not LABEL_RX.search(m.group(1)):
                out.append(env)
    return out

def _task_ids_from_plan(plan_path: Path) -> List[str]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    return [t["id"] for t in plan.get("tasks", [])]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snippets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--apply_todos", action="store_true")
    ap.add_argument("--plan", default=None, help="Optional plan.json to restrict checks to current tasks.")
    args = ap.parse_args()

    sdir = Path(args.snippets)
    targets: List[Path]
    if args.plan:
        ids = _task_ids_from_plan(Path(args.plan))
        targets = [sdir / f"{tid}.tex" for tid in ids]
    else:
        targets = sorted(sdir.glob("*.tex"))

    report: Dict[str, Any] = {"files": {}}
    for p in targets:
        if not p.exists():
            # may be PREAMBLE/TITLE-only plans
            continue
        txt = p.read_text(encoding="utf-8")
        issues = {
            "units_raw": detect_units(txt),
            "needs_align": needs_align(txt),
            "missing_labels": missing_labels(txt),
        }
        report["files"][p.name] = issues

        if args.apply_todos:
            todos: List[str] = []
            if issues["needs_align"]:
                todos.append("\\todo{Consider using an align/aligned environment for multi-line equations with '='.}")
            if issues["units_raw"]:
                todos.append("\\todo{Wrap numeric+unit in \\SI{...}{...} or \\si{...}.}")
            if issues["missing_labels"]:
                todos.append("\\todo{Add \\label{} inside each figure/table/equation.}")
            if todos:
                p.write_text("% Postflight notes\n" + "\n".join(todos) + "\n\n" + txt, encoding="utf-8")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[postflight] wrote {args.out}")

if __name__ == "__main__":
    main()
