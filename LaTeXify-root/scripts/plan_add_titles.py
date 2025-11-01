#!/usr/bin/env python3
from __future__ import annotations
import json, argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=False)
    args = ap.parse_args()
    p = Path(args.plan)
    plan = json.loads(p.read_text(encoding="utf-8"))
    changed = False

    for i, t in enumerate(plan.get("tasks", []), 1):
        if "title" not in t or not t["title"]:
            # Heuristic: TITLE stays "Title", otherwise Qn or Section n
            tid = t.get("id", f"T{i}")
            if tid.upper() == "TITLE":
                t["title"] = "Title"
            else:
                t["title"] = tid
            changed = True

    outp = Path(args.out) if args.out else p
    outp.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[plan] wrote → {outp} (changed={changed})")

if __name__ == "__main__":
    main()
