# scripts/classes/qa_agent.py
from __future__ import annotations
import argparse, json
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description="Class QA stub over a class probe report.")
    ap.add_argument("--report", required=True, help="Path to probe_report.json")
    args = ap.parse_args()

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    cls = report.get("class")
    ok = report.get("ok")
    issues = report.get("issues", [])
    suggestions = report.get("suggestions", {})

    print("# Class QA Summary")
    print(f"- class: {cls}")
    print(f"- ok: {ok}")
    print("- issues:")
    for it in issues:
        print(f"  - {it['type']}: {it.get('name','')}")
    print("- next-steps (fedora):")
    for s in suggestions.get("fedora", []):
        print(f"  - {s}")
    print("- notes:")
    for n in suggestions.get("notes", []):
        print(f"  - {n.get('action','note')}: {n}")

if __name__ == "__main__":
    main()
