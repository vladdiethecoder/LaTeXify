#!/usr/bin/env python3
"""
fix_merge_conflicts.py

Quickly strip Git conflict markers from a file by keeping the desired side.

Usage:
  python scripts/fix_merge_conflicts.py dev/ocr_backends/nanonets_ocr2.py --prefer head
  python scripts/fix_merge_conflicts.py dev/ocr_backends/nanonets_ocr2.py --prefer other

This is a mechanical fix to get the file compiling again. You should still
review the result. The script writes a backup next to the file: <path>.bak
"""
from __future__ import annotations

import argparse
from pathlib import Path

START = "<<<<<<<"
MID   = "======="
END   = ">>>>>>>"

def fix_file(path: Path, prefer: str) -> None:
    src = path.read_text(encoding="utf-8").splitlines()
    out, buf_head, buf_other = [], [], []
    in_block, in_other = False, False
    for line in src:
        if line.startswith(START):
            in_block, in_other = True, False
            buf_head, buf_other = [], []
            continue
        if in_block and line.startswith(MID):
            in_other = True
            continue
        if in_block and line.startswith(END):
            chosen = buf_head if prefer == "head" else buf_other
            out.extend(chosen)
            in_block, in_other = False, False
            buf_head, buf_other = [], []
            continue
        if in_block:
            (buf_other if in_other else buf_head).append(line)
        else:
            out.append(line)
    backup = path.with_suffix(path.suffix + ".bak")
    backup.write_text("\n".join(src) + "\n", encoding="utf-8")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"[fix] Wrote cleaned file: {path} (backup: {backup})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    ap.add_argument("--prefer", choices=["head", "other"], default="head",
                    help="Which side to keep from conflict blocks.")
    args = ap.parse_args()
    fix_file(args.path, args.prefer)

if __name__ == "__main__":
    main()
