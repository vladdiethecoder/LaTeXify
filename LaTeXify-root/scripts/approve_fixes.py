#!/usr/bin/env python3
"""Approve and promote successful auto-fixes into the LaTeX knowledge base."""

from __future__ import annotations

import argparse
#!/usr/bin/env python3
"""Approve and promote successful auto-fixes into the LaTeX knowledge base."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List

HERE = Path(__file__).resolve().parent
DEFAULT_FIXES = HERE.parent / "build" / "successful_fixes.jsonl"
DEFAULT_KB = HERE.parent / "kb" / "latex" / "latex_docs.jsonl"
DEFAULT_INDEX_RUN = HERE / "build_index.py"


def read_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    out: List[Dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def append_jsonl(path: Path, records: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def build_kb_entry(record: Dict) -> Dict:
    error = record.get("error", {})
    snippet_path = record.get("snippet")
    capabilities = record.get("auto_fix", {}).get("capabilities") or []
    answer_lines = [
        f"This entry documents an automated fix for the LaTeX error '{error.get('message', 'unknown')}'.",
        "",
        "Steps applied:",
        "- Updated the snippet in place with an auto-fix comment.",
    ]
    if capabilities:
        answer_lines.append(f"- Declared capabilities: {', '.join(capabilities)}.")
    answer_lines.append("- Verified compilation succeeded after the change.")
    answer_lines.append("")
    answer_lines.append("Updated snippet:")
    answer_lines.append(record.get("after", ""))

    return {
        "source": "auto_fix_agent",
        "url": f"snippet://{snippet_path}",
        "title": f"Auto fix: {error.get('category', 'unknown')} at {snippet_path}",
        "tags": [error.get("category", "auto_fix")],
        "question": f"How was the error '{error.get('message', 'unknown')}' resolved?",
        "answer": "\n".join(answer_lines).strip(),
        "code_blocks": [record.get("after", "")],
        "notes": "approved_auto_fix",
        "metadata": {
            "snippet": snippet_path,
            "error": error,
            "before": record.get("before", ""),
        },
    }


def prompt_yes(question: str, default: bool = False) -> bool:
    prompt = " [Y/n]" if default else " [y/N]"
    try:
        resp = input(question + prompt + " ").strip().lower()
    except EOFError:
        return default
    if not resp:
        return default
    return resp in {"y", "yes"}


def rebuild_index(index_script: Path, kb_dir: Path) -> None:
    try:
        subprocess.run(
            [sys.executable, str(index_script), "--run_dir", str(kb_dir)],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        print(f"[approve_fixes] Index rebuild failed: {exc}", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description="Promote successful auto-fixes into the KB")
    ap.add_argument("--fixes", type=Path, default=DEFAULT_FIXES)
    ap.add_argument("--kb", type=Path, default=DEFAULT_KB)
    ap.add_argument("--index-script", type=Path, default=DEFAULT_INDEX_RUN)
    ap.add_argument("--yes", action="store_true", help="Approve all fixes without prompting")
    ap.add_argument("--rebuild-index", action="store_true", help="Rebuild FAISS index after approval")
    args = ap.parse_args()

    fixes = read_jsonl(args.fixes)
    if not fixes:
        print(f"No fixes found in {args.fixes}")
        return 0

    approved: List[Dict] = []
    for rec in fixes:
        error = rec.get("error", {})
        print("\n=== Auto Fix Candidate ===")
        print(f"Snippet: {rec.get('snippet')}")
        print(f"Error:   {error.get('category')} :: {error.get('message')}")
        print("--- Before ---")
        print(rec.get("before", ""))
        print("--- After ---")
        print(rec.get("after", ""))
        if not args.yes and not prompt_yes("Approve this fix?", default=False):
            continue
        approved.append(build_kb_entry(rec))

    if not approved:
        print("No fixes approved.")
        return 0

    append_jsonl(args.kb, approved)
    print(f"Appended {len(approved)} entries to {args.kb}")

    if args.rebuild_index:
        rebuild_index(args.index_script, args.kb.parent)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
