#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Concatenate snippet files into build/main.tex
"""
from __future__ import annotations
import argparse, json, pathlib

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--snippets", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    plan = json.load(open(args.plan, "r"))
    out = pathlib.Path(args.out)
    txt = out.read_text(encoding="utf-8")  # already has preamble+title and \end{document}

    # Insert questions just before \end{document}
    marker = "\\end{document}"
    head, _, tail = txt.partition(marker)
    lines = [head.rstrip(), ""]

    for qid in plan.get("questions", []):
        qpath = pathlib.Path(args.snippets) / f"{qid}.tex"
        if qpath.exists():
            lines.append(f"\\section*{{{qid}}}")
            lines.append(qpath.read_text(encoding="utf-8").strip())
            lines.append("")

    final = "\n".join(lines) + "\n" + marker + "\n"
    out.write_text(final, encoding="utf-8")
    print(f"Wrote main.tex → {out}")

if __name__ == "__main__":
    main()
