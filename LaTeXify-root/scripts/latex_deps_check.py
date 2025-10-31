#!/usr/bin/env python3
"""
Parse LaTeX logs for missing .sty files and print install suggestions.

Deterministic, no network, emits a single JSON object:
{
  "missing": ["GS1.sty", ...],
  "suggest": {
    "dnf": ["sudo dnf install texlive-gs1", ...],
    "tlmgr": ["tlmgr install gs1", ...]
  }
}
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

# Minimal, curated map for Fedora (DNF) + TeX Live (tlmgr)
# Keys must be .sty filenames exactly as LaTeX reports them.
# Values: (dnf_pkg, tlmgr_pkg)
KNOWN: Dict[str, Tuple[str, str]] = {
    # previously-seen
    "thmtools.sty": ("texlive-thmtools", "thmtools"),
    "lastpage.sty": ("texlive-lastpage", "lastpage"),
    "svg.sty": ("texlive-svg", "svg"),
    # new in this increment
    "GS1.sty": ("texlive-gs1", "gs1"),
}

_MISSING_RE = re.compile(r"File `([^']+\.sty)' not found")

def parse_missing_styles(text: str) -> List[str]:
    """Return sorted unique .sty names reported missing in a LaTeX log."""
    found: Set[str] = set(m.group(1) for m in _MISSING_RE.finditer(text))
    return sorted(found)

def suggest_installs(stys: Iterable[str]) -> Dict[str, List[str]]:
    """Map missing .sty files to platform-specific install commands."""
    dnf_cmds: List[str] = []
    tlmgr_cmds: List[str] = []

    for sty in sorted(set(stys)):
        if sty in KNOWN:
            dnf_pkg, tl_pkg = KNOWN[sty]
            dnf_cmds.append(f"sudo dnf install {dnf_pkg}")
            tlmgr_cmds.append(f"tlmgr install {tl_pkg}")
        else:
            # Generic fallbacks (works with Fedora virtual provides and tlmgr)
            dnf_cmds.append(f"sudo dnf install 'tex({sty})'")
            tlmgr_cmds.append(f"tlmgr install {Path(sty).stem}")

    # Stable, deterministic order
    return {"dnf": dnf_cmds, "tlmgr": tlmgr_cmds}

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Detect missing LaTeX packages from a .log and suggest installs (deterministic)."
    )
    ap.add_argument("--from-log", type=Path, help="Path to LaTeX build log (e.g., build/main.log)")
    ap.add_argument("--print-mapping", action="store_true", help="Print known map and exit")
    args = ap.parse_args()

    if args.print_mapping:
        print(json.dumps({"known": sorted(KNOWN.keys())}, ensure_ascii=False))
        return

    if not args.from_log or not args.from_log.exists():
        print(json.dumps({"error": "log_not_found", "path": str(args.from_log) if args.from_log else None}))
        sys.exit(2)

    text = args.from_log.read_text(encoding="utf-8", errors="ignore")
    missing = parse_missing_styles(text)
    suggestions = suggest_installs(missing)

    out = {"missing": missing, "suggest": suggestions}
    print(json.dumps(out, ensure_ascii=False))
    # exit code 0 even if missing; this is advisory
    sys.exit(0)

if __name__ == "__main__":
    main()
