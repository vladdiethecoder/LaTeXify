#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
fix_snippets.py (v1.2.1 hot-fix)

Automated cleanup for LaTeX snippets to meet rubric & preamble policy:

1) Units -> siunitx v3:
   Wrap bare "number + unit" (e.g., `125 m`, `50 km`, `3.5 s`) into
   `\qty{<num>}{<unit>}`. Existing \qty/\SI/\unit are left untouched.

2) Equation alignment:
   If an equation block looks like a derivation (multi-line or multiple '='),
   convert `equation(*)` -> `align` and align first '=' via '&=' per line.

3) Labels and refs:
   - Ensure a `\label{...}` exists in each equation/align environment using a
     deterministic content-hash label (eq:<8-hex>).
   - Convert `\ref{...}` -> `\cref{...}` (preserving \Cref/\cref already present).

4) Strip previous `\todo{...}` lines so the final PDF is clean.

Idempotent: running multiple times yields the same output.

Usage:
  python scripts/fix_snippets.py \
    --snippets build/snippets \
    --plan build/plan.json
"""
from __future__ import annotations

import re
import json
import hashlib
import argparse
from pathlib import Path
from typing import List, Tuple


# --- Regexes ---------------------------------------------------------------

# \ref -> \cref (avoid touching \cref/\Cref already)
REF_RX   = re.compile(r'(?<!\\c)\\ref\{([^}]+)\}')

# remove explicit todos and any postflight banner lines
TODO_RX  = re.compile(r'\\todo\{[^}]*\}')
POSTFLT  = re.compile(r'^\s*%+\s*Postflight.*$', re.IGNORECASE)

# simple "number + unit" captures (not within \qty/\SI/\num already)
UNIT_TOKENS = [
    # lengths
    r"m", r"cm", r"mm", r"km",
    # time
    r"s", r"ms", r"h",
    # mass
    r"kg", r"g",
    # derived/common
    r"N", r"J", r"W", r"Pa", r"Hz",
    # simple ratios we often see inline
    r"m/s", r"km/h",
]
# Guard against matching when already inside \qty{..}{..} or \SI{..}{..}
UNIT_RX = re.compile(
    rf'(?<!\\qty\{{)(?<!\\SI\{{)(?<!\\num\{{)'
    rf'(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>(?:{"|".join(UNIT_TOKENS)}))\b'
)

# equation environments (exclude align/aligned which we keep)
EQN_BLOCK = re.compile(
    r'\\begin\{equation\*?\}(?P<body>.*?)\\end\{equation\*?\}',
    re.S
)
LABEL_RX = re.compile(r'\\label\{[^}]+\}')
ALIGN_OR_ALIGNED = re.compile(r'\\begin\{(align\*?|aligned)\}')


# --- Helpers ----------------------------------------------------------------

def _hash_label(prefix: str, body: str) -> str:
    h = hashlib.sha1(body.encode("utf-8")).hexdigest()[:8]
    return f"{prefix}:{h}"

def _strip_todos(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if TODO_RX.search(line):
            continue
        if POSTFLT.search(line):
            continue
        lines.append(line)
    return "\n".join(lines)

def _units_to_qty(text: str) -> str:
    def _repl(m: re.Match) -> str:
        num = m.group("num")
        unit = m.group("unit")
        # Use literal unit in the second arg: siunitx parses these as units.
        # (Users can later convert to macro units like \metre if desired.)
        return fr"\qty{{{num}}}{{{unit}}}"
    return UNIT_RX.sub(_repl, text)

def _ensure_label_in_env(body: str, prefix: str = "eq") -> str:
    if LABEL_RX.search(body):
        return body
    label = _hash_label(prefix, body)
    if not body.endswith("\n"):
        body += "\n"
    return body + fr"\label{{{label}}}"

def _convert_equations_to_align(text: str) -> str:
    """Convert derivation-like equation blocks to align and ensure labels."""
    def _convert_block(m: re.Match) -> str:
        body = m.group("body")
        multiline = "\n" in body
        eq_count = body.count("=")
        if not multiline and eq_count < 2:
            # Keep as equation but ensure label
            labeled = _ensure_label_in_env(body, "eq")
            return f"\\begin{{equation}}\n{labeled}\n\\end{{equation}}"

        # Convert to align; align first '=' per line if not already set
        lines = body.splitlines()
        new_lines: List[str] = []
        for ln in lines:
            if "&=" in ln or "\\\\" in ln:
                new_lines.append(ln)
                continue
            if "=" in ln:
                new_lines.append(ln.replace("=", "&=", 1))
            else:
                new_lines.append(ln)
        aligned = "\n".join(new_lines)
        aligned = _ensure_label_in_env(aligned, "eq")
        return f"\\begin{{align}}\n{aligned}\n\\end{{align}}"

    return EQN_BLOCK.sub(_convert_block, text)

def _refs_to_cref(text: str) -> str:
    return REF_RX.sub(r'\\cref{\1}', text)


def fix_snippet_text(text: str) -> str:
    # Order matters: strip -> units -> refs -> equations
    text0 = _strip_todos(text)
    text1 = _units_to_qty(text0)
    text2 = _refs_to_cref(text1)
    text3 = _convert_equations_to_align(text2)
    return text3


# --- CLI --------------------------------------------------------------------

def _task_ids_from_plan(plan_path: Path) -> List[str]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    return [t["id"] for t in plan.get("tasks", [])]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snippets", required=True)
    ap.add_argument("--plan", required=True)
    args = ap.parse_args()

    sdir = Path(args.snippets)
    task_ids = _task_ids_from_plan(Path(args.plan))

    changed = 0
    for tid in task_ids:
        p = sdir / f"{tid}.tex"
        if not p.exists():
            continue
        old = p.read_text(encoding="utf-8")
        new = fix_snippet_text(old)
        if new != old:
            p.write_text(new, encoding="utf-8")
            changed += 1

    print(f"[fix] processed {len(task_ids)} targets; changed {changed} files")

if __name__ == "__main__":
    main()
