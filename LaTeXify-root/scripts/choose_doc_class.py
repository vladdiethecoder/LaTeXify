#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

LOG = Path("build/choose_doc_class.log.jsonl")

def _log(event: str, **details) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"event": event, **details}, ensure_ascii=False) + "\n")

def main() -> None:
    ap = argparse.ArgumentParser(description="Patch plan.json to use a specific doc_class")
    ap.add_argument("--plan", default="plan.json")
    ap.add_argument("--doc_class", required=True, help="e.g., textbook, paper, thesis, novel, news, poem, ieee_modern")
    args = ap.parse_args()

    plan_p = Path(args.plan)
    if not plan_p.exists():
        raise SystemExit(f"Plan not found: {plan_p}")

    plan = json.loads(plan_p.read_text(encoding="utf-8"))
    before = plan.get("doc_class")
    plan["doc_class"] = args.doc_class
    plan_p.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    _log("doc_class_patched", plan=str(plan_p), before=before, after=args.doc_class)
    print(json.dumps({"plan": str(plan_p), "doc_class": args.doc_class}))
if __name__ == "__main__":
    main()
