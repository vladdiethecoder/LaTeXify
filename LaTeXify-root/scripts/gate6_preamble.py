#!/usr/bin/env python
import os, argparse, textwrap
p = argparse.ArgumentParser()
p.add_argument("--out", default="runs/sample/main.tex")
p.add_argument("--snippet", default="runs/sample/planner_smoke.tex")
args = p.parse_args()

preamble = r"""\documentclass{lix_article}
\usepackage{amsmath, amssymb, mathtools}
\usepackage{thmtools}
\usepackage{cleveref}
\usepackage{graphicx}
\usepackage{minted} % requires -shell-escape
\usepackage{xcolor}
\title{LaTeXify Smoke Test}
\author{Auto}
\begin{document}
\maketitle
\tableofcontents
\section{Introduction}
This is a smoke test. If any OCR doubts exist, we insert \verb|\todo{}|.
"""

tail = r"""
\section{Code Listing (minted)}
\begin{minted}{python}
def f(x): return x**2
\end{minted}
\end{document}
"""

os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "w", encoding="utf-8") as w:
    w.write(preamble)
    if os.path.exists(args.snippet):
        with open(args.snippet, "r", encoding="utf-8") as s:
            w.write("\n% --- inserted from planner ---\n")
            w.write(s.read())
    w.write(tail)
print("WROTE", args.out)
