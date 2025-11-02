from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import List, Dict
from latexify.kb.kb_paths import (
    REQUIRED_INDEX_FILES,
    ensure_dir,
    find_first_with_index,
    has_index,
    link_or_copy,
)

DEFAULT_ALIAS = Path("kb/latex")
DEFAULT_SOURCES = [Path("kb/latex"), Path("kb/offline/latex"), Path("kb/online/latex")]

def _copy_optional(src_dir: Path, alias_dir: Path):
    staged = []
    for name in ("latex_docs.jsonl", "chunks.jsonl"):
        s = src_dir / name
        if s.exists():
            action, dst = link_or_copy(s, alias_dir / name)
            staged.append({"file": name, "action": action, "dst": dst})
    return staged

def ensure_kb_alias(alias_dir: Path, candidates: List[Path]) -> Dict:
    alias_dir = alias_dir.resolve()
    ensure_dir(alias_dir)
    if has_index(alias_dir):
        return {"ok": True, "alias_dir": str(alias_dir), "resolved_from": "self",
                "staged": [{"file": f, "action": "exists", "dst": str(alias_dir / f)} for f in REQUIRED_INDEX_FILES],
                "payloads": _copy_optional(alias_dir, alias_dir)}
    src = find_first_with_index([c.resolve() for c in candidates if c.exists()])
    if not src:
        return {"ok": False, "alias_dir": str(alias_dir),
                "error": f"No FAISS index found in candidates: {[str(c) for c in candidates]}"}
    staged = []
    for f in REQUIRED_INDEX_FILES:
        action, dst = link_or_copy(src / f, alias_dir / f)
        staged.append({"file": f, "action": action, "dst": dst})
    payloads = _copy_optional(src, alias_dir)
    return {"ok": True, "alias_dir": str(alias_dir), "resolved_from": str(src), "staged": staged, "payloads": payloads}

def main():
    ap = argparse.ArgumentParser(description="Ensure canonical KB alias kb/latex exists (symlink/copy from offline/online).")
    ap.add_argument("--alias_dir", type=Path, default=DEFAULT_ALIAS)
    ap.add_argument("--source", action="append", default=[], help="Add source dirs to consider (priority order).")
    args = ap.parse_args()
    sources = [Path(s) for s in (args.source or [str(p) for p in DEFAULT_SOURCES])]
    report = ensure_kb_alias(args.alias_dir, sources)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report.get("ok"):
        raise SystemExit(1)

if __name__ == "__main__":
    main()
