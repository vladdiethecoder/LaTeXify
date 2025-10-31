#!/usr/bin/env python3
"""
scripts/check_indexes.py

Prints a quick status table for expected indexes:
  indexes/{assignment,assessment,rubric,user}/(faiss.index, faiss.meta.json)
  indexes/latex_docs.(index, meta.json)

Exit code 0 if all present, else 1.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Tuple

EXPECTED = [
    ("assignment", "indexes/assignment/faiss.index", "indexes/assignment/faiss.meta.json"),
    ("assessment", "indexes/assessment/faiss.index", "indexes/assessment/faiss.meta.json"),
    ("rubric",     "indexes/rubric/faiss.index",     "indexes/rubric/faiss.meta.json"),
    ("user",       "indexes/user/faiss.index",       "indexes/user/faiss.meta.json"),
    ("latex_docs", "indexes/latex_docs.index",       "indexes/latex_docs.meta.json"),
]


def main() -> None:
    ok = True
    rows: List[Tuple[str, str]] = []
    for name, idx, meta in EXPECTED:
        idx_p, meta_p = Path(idx), Path(meta)
        s = "OK" if idx_p.exists() and meta_p.exists() else "MISSING"
        if s == "MISSING":
            ok = False
        rows.append((name, s))
    w = max(len(n) for n, _ in rows) + 2
    print("Index status:")
    for n, s in rows:
        print(f"  {n:<{w}} {s}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
