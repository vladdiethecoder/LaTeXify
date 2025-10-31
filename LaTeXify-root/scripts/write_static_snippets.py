#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Write PREAMBLE.tex and TITLE.tex into build/snippets.
Correct package order: hyperref BEFORE cleveref (required).  # See: Cref after hyperref
"""
from __future__ import annotations
import argparse, json, pathlib

def build_preamble(doc_class: str) -> str:
    # hyperref before cleveref (see docs)
    return r"""
% --- AUTOGEN PREAMBLE (do not edit by hand) ---
\documentclass{""" + doc_class + r"""}

% Math & theorem stack
\usepackage{amsmath,amssymb,amsthm,mathtools}
\usepackage{thmtools}

% Graphics, tables, etc.
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{caption}
\usepackage{siunitx}
\usepackage{array}
\usepackage{microtype}
\usepackage{enumitem}
\usepackage[margin=1in]{geometry}

% Hyperlinks FIRST, cleveref AFTER (required)
\usepackage[hidelinks]{hyperref}
\usepackage{cleveref} % must be loaded after hyperref

% Optional todos (safe fallback if package missing)
\IfFileExists{todonotes.sty}{\usepackage{todonotes}}{\newcommand{\todo}[1]{\textbf{[TODO: ##1]}}}

\begin{document}
"""

def build_title(title: str, author: str, course: str, date: str) -> str:
    return rf"""
\title{{{title}}}
\author{{{author} \\ \small {course}}}
\date{{{date}}}
\maketitle
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--snippets", required=True)
    ap.add_argument("--main", required=True)
    args = ap.parse_args()

    plan = json.load(open(args.plan, "r"))
    snip_dir = pathlib.Path(args.snippets)
    snip_dir.mkdir(parents=True, exist_ok=True)

    preamble = build_preamble(plan["doc_class"])
    title = build_title(plan["title"], plan["author"], plan["course"], plan["date"])

    (snip_dir / "PREAMBLE.tex").write_text(preamble.strip() + "\n", encoding="utf-8")
    (snip_dir / "TITLE.tex").write_text(title.strip() + "\n", encoding="utf-8")

    # write minimal main.tex skeleton; aggregation script will append Q*.tex
    pathlib.Path(args.main).write_text(
        "\\input{snippets/PREAMBLE.tex}\n"
        "\\input{snippets/TITLE.tex}\n"
        "% questions appended by aggregate_tex.py\n"
        "\\end{document}\n",
        encoding="utf-8"
    )
    print("Wrote build-aware snippets: PREAMBLE.tex + TITLE.tex")

if __name__ == "__main__":
    main()
