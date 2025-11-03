#!/usr/bin/env python3
"""Promote auto-fix logs into the LaTeX knowledge base and rebuild the index."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
DEFAULT_LOG = PROJECT_ROOT / "build" / "successful_fixes.jsonl"
DEFAULT_DATA = PROJECT_ROOT / "data" / "latex_docs.jsonl"
DEFAULT_KB_DIR = PROJECT_ROOT / "kb" / "latex"
REGEN_SCRIPT = HERE / "regen_chunks_from_docs.py"
BUILD_INDEX_SCRIPT = HERE / "build_index.py"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def append_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_existing_keys(path: Path) -> Set[str]:
    keys: Set[str] = set()
    if not path.exists():
        return keys
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            meta = rec.get("metadata")
            if isinstance(meta, dict):
                key = meta.get("auto_fix_key")
                if key:
                    keys.add(str(key))
    return keys


def compute_auto_fix_key(record: Dict[str, Any]) -> Optional[str]:
    snippet = record.get("snippet") or ""
    error = record.get("error") or {}
    message = str(error.get("message") or "")
    category = str(error.get("category") or "")
    line = error.get("line")
    signature = f"{category}::{message}::{line}"
    final_snippet = record.get("final_snippet") or record.get("after") or ""
    preamble = "\n".join(record.get("preamble_additions") or record.get("auto_fix", {}).get("preamble_additions") or [])
    payload = "::".join([str(snippet), signature, final_snippet, preamble])
    if not payload.strip():
        return None
    digest = hashlib.sha1(payload.encode("utf-8", errors="ignore")).hexdigest()
    return f"{snippet}::{digest}"


def build_entry(record: Dict[str, Any], key: str) -> Dict[str, Any]:
    error = record.get("error") or {}
    snippet_path = str(record.get("snippet") or "unknown")
    final_snippet = record.get("final_snippet") or record.get("after") or ""
    preamble_additions: List[str] = []
    for raw in record.get("preamble_additions") or record.get("auto_fix", {}).get("preamble_additions") or []:
        text = str(raw).strip()
        if text:
            preamble_additions.append(text)
    capabilities = record.get("auto_fix", {}).get("capabilities") or []
    what_changed = record.get("auto_fix", {}).get("what_changed")

    tags = sorted({t for t in (error.get("category"), "auto_fix") if t})
    answer_lines = [
        f"This snippet resolves the LaTeX error '{error.get('message', 'unknown')}'.",
        "",
        "Automatic fix summary:",
        f"- Error category: {error.get('category', 'unknown')}",
        f"- Target snippet: {snippet_path}",
    ]
    if what_changed:
        answer_lines.append(f"- Action taken: {what_changed}")
    if capabilities:
        answer_lines.append(f"- Declared capabilities: {', '.join(capabilities)}")
    if preamble_additions:
        answer_lines.append("- Preamble additions:")
        for addition in preamble_additions:
            answer_lines.append(f"  * {addition}")
    answer_lines.append("- Verified compilation succeeded after applying this change.")
    answer_lines.append("")
    answer_lines.append("Final snippet:")
    answer_lines.append(final_snippet.strip() or "(no snippet diff recorded)")

    code_blocks: List[str] = []
    if final_snippet.strip():
        code_blocks.append(final_snippet.strip())
    for addition in preamble_additions:
        if addition not in code_blocks:
            code_blocks.append(addition)

    entry = {
        "source": "auto_fix_agent",
        "url": f"snippet://{snippet_path}",
        "title": f"Auto-fix: {error.get('category', 'unknown')} at {snippet_path}",
        "tags": tags,
        "question": f"How was the LaTeX error '{error.get('message', 'unknown')}' resolved?",
        "answer": "\n".join(answer_lines).strip(),
        "code_blocks": code_blocks,
        "notes": "auto_fix_promoted",
        "metadata": {
            "auto_fix_key": key,
            "snippet": snippet_path,
            "error": error,
            "timestamp": record.get("timestamp"),
            "preamble_additions": preamble_additions,
            "kb_suggestions": record.get("kb_suggestions") or record.get("auto_fix", {}).get("kb_suggestions"),
        },
    }
    return entry


def sync_kb_docs(data_path: Path, kb_dir: Path) -> Path:
    kb_dir.mkdir(parents=True, exist_ok=True)
    kb_path = kb_dir / "latex_docs.jsonl"
    if kb_path.exists() or kb_path.is_symlink():
        kb_path.unlink(missing_ok=True)
    shutil.copyfile(data_path, kb_path)
    return kb_path


def rebuild_indexes(kb_dir: Path) -> None:
    subprocess.run(
        [sys.executable, str(REGEN_SCRIPT), "--docs", str(kb_dir / "latex_docs.jsonl"), "--out_dir", str(kb_dir)],
        check=True,
    )
    subprocess.run(
        [sys.executable, str(BUILD_INDEX_SCRIPT), "--run_dir", str(kb_dir)],
        check=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Promote successful auto-fixes into latex_docs.jsonl and rebuild indexes")
    ap.add_argument("--log", type=Path, default=DEFAULT_LOG, help="Path to successful_fixes.jsonl")
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Destination latex_docs.jsonl file")
    ap.add_argument("--kb-dir", type=Path, default=DEFAULT_KB_DIR, help="KB directory to mirror data into")
    ap.add_argument("--dry-run", action="store_true", help="Preview promotions without writing files or rebuilding indexes")
    args = ap.parse_args()

    records = read_jsonl(args.log)
    if not records:
        print(f"[promote_fixes] No successful fixes found in {args.log}")
        return 0

    existing_keys = load_existing_keys(args.data)
    staged: List[Dict[str, Any]] = []
    staged_keys: Set[str] = set()
    for rec in records:
        key = compute_auto_fix_key(rec)
        if not key or key in existing_keys or key in staged_keys:
            continue
        entry = build_entry(rec, key)
        staged.append(entry)
        staged_keys.add(key)

    if not staged:
        print("[promote_fixes] No new fixes to promote.")
        return 0

    print(f"[promote_fixes] Promoting {len(staged)} fixes into {args.data}")
    if args.dry_run:
        for entry in staged:
            print(json.dumps(entry, ensure_ascii=False, indent=2))
        return 0

    append_jsonl(args.data, staged)
    kb_path = sync_kb_docs(args.data, args.kb_dir)
    rebuild_indexes(args.kb_dir)
    print(f"[promote_fixes] Updated {args.data} and mirrored to {kb_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
