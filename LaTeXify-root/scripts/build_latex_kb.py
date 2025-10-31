#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_latex_kb.py

Phase 0: Produce a small, deterministic LaTeX mechanics knowledge base for
typesetting "how-tos" used by the Synth stage (aligned equations, theorem envs,
cleveref, siunitx, booktabs tables, figures, etc.).

Outputs:
  - data/latex_docs.jsonl         (human-readable KB)
  - dev/runs/latex_kb/chunks.jsonl (same content as chunks)
  - indexes/latex_docs.index/      (FAISS, via build_index_bgem3.py if requested)

Usage:
  python scripts/build_latex_kb.py
  python scripts/build_latex_kb.py --no_index        # only write JSONL/chunks
  python scripts/build_latex_kb.py --run_dir dev/runs/latex_kb --out indexes/latex_docs.index
"""
from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import List, Dict

DEFAULT_RUN_DIR = "dev/runs/latex_kb"
DEFAULT_OUT_DIR = "indexes/latex_docs.index"
DEFAULT_DATA    = "data/latex_docs.jsonl"

# Minimal, deterministic KB entries (mechanics only; safe to embed)
KB_ENTRIES: List[Dict] = [
    {
        "id": "align-equations",
        "topic": "math-align",
        "title": "Aligned Equations with amsmath",
        "source": "builtin",
        "text": (
            "Use the amsmath 'align' environment to align on '='. "
            "Label key steps with \\label{} and refer with \\cref{}."
        ),
        "latex": (
            "\\begin{align}\n"
            "  a + b &= c \\label{eq:abc}\\\\\n"
            "  d &= e + f\n"
            "\\end{align}\n"
            "As shown in \\cref{eq:abc}."
        )
    },
    {
        "id": "theorem-thmtools",
        "topic": "thmtools",
        "title": "Definitions & Theorems (thmtools)",
        "source": "builtin",
        "text": (
            "Use thmtools to define theorem-like environments. "
            "Number within sections for clarity."
        ),
        "latex": (
            "\\declaretheorem[name=Definition,numberwithin=section]{definition}\n"
            "\\begin{definition}[Vector Space]\n"
            "A set V with operations + and scalar multiplication satisfying axioms.\n"
            "\\end{definition}\n"
        )
    },
    {
        "id": "cleveref-refs",
        "topic": "cleveref",
        "title": "Cross-References with cleveref",
        "source": "builtin",
        "text": "Use \\label and \\cref for automatic prefixing (Eq., Def., Fig.).",
        "latex": "See \\cref{eq:abc,fig:diagram} for references."
    },
    {
        "id": "siunitx-units",
        "topic": "siunitx",
        "title": "Units with siunitx",
        "source": "builtin",
        "text": "Use \\SI{<number>}{<unit>} and \\si{<unit>} for units.",
        "latex": "Velocity is \\SI{3.0}{\\meter\\per\\second} and g \\approx \\SI{9.81}{\\meter\\per\\second\\squared}."
    },
    {
        "id": "graphics-figure",
        "topic": "graphicx",
        "title": "Figures with graphicx",
        "source": "builtin",
        "text": "Use figure environment, label below caption. Use \\centering.",
        "latex": (
            "\\begin{figure}[h]\n"
            "\\centering\n"
            "\\includegraphics[width=0.6\\textwidth]{example-image}\n"
            "\\caption{Demo figure}\\label{fig:diagram}\n"
            "\\end{figure}"
        )
    },
    {
        "id": "booktabs-table",
        "topic": "booktabs",
        "title": "Tables with booktabs",
        "source": "builtin",
        "text": "Use top/mid/bottom rules; avoid vertical lines.",
        "latex": (
            "\\begin{table}[h]\n"
            "\\centering\n"
            "\\begin{tabular}{ll}\n"
            "\\toprule\n"
            "Quantity & Value\\\\\\midrule\n"
            "Mass & \\SI{1.0}{\\kilogram}\\\\\n"
            "Time & \\SI{2.0}{\\second}\\\\\\bottomrule\n"
            "\\end{tabular}\n"
            "\\caption{Example table}\\label{tab:example}\n"
            "\\end{table}"
        )
    },
    {
        "id": "enumitem-lists",
        "topic": "enumitem",
        "title": "Lists with enumitem",
        "source": "builtin",
        "text": "Customize labels and spacing with enumitem.",
        "latex": "\\begin{itemize}[left=*,nosep]\\item First\\item Second\\end{itemize}"
    },
    {
        "id": "geometry-margins",
        "topic": "geometry",
        "title": "Margins with geometry",
        "source": "builtin",
        "text": "Set 1in margins for readability.",
        "latex": "\\usepackage[margin=1in]{geometry}"
    },
    {
        "id": "microtype",
        "topic": "microtype",
        "title": "Microtypography",
        "source": "builtin",
        "text": "Improve justification and kerning.",
        "latex": "\\usepackage{microtype}"
    },
    {
        "id": "hyperref",
        "topic": "hyperref",
        "title": "Links with hyperref",
        "source": "builtin",
        "text": "Enable PDF metadata and links.",
        "latex": "\\usepackage[hidelinks]{hyperref}"
    },
]


def save_jsonl(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default=DEFAULT_RUN_DIR)
    ap.add_argument("--out", default=DEFAULT_OUT_DIR)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--no_index", action="store_true", help="Only write JSONL/chunks; skip indexing.")
    args = ap.parse_args()

    # 1) Write KB to data/latex_docs.jsonl
    data_path = Path(args.data)
    save_jsonl(data_path, KB_ENTRIES)

    # 2) Convert to chunks.jsonl under run_dir
    chunks = []
    for i, row in enumerate(KB_ENTRIES):
        text = f"{row['title']}\n\n{row['text']}\n\n{row['latex']}"
        chunks.append({
            "id": f"kb-{row['id']}",
            "text": text,
            "page": None,
            "bbox": None,
            "block_type": "KB",
            "source_backend": "builtin",
            "semantic_id": row["topic"],
            "flags": {"low_confidence": False, "high_ocr_disagreement": False}
        })
    save_jsonl(Path(args.run_dir) / "chunks.jsonl", chunks)
    (Path(args.run_dir) / "chunks_meta.json").write_text(
        json.dumps({"count": len(chunks)}, indent=2), encoding="utf-8"
    )
    print(f"[latex_kb] wrote {data_path}")
    print(f"[latex_kb] wrote {Path(args.run_dir) / 'chunks.jsonl'}")

    # 3) Optionally index via build_index_bgem3.py
    if not args.no_index:
        import subprocess, sys
        cmd = [
            sys.executable, "scripts/build_index_bgem3.py",
            "--run_dir", args.run_dir,
            "--out", args.out
        ]
        print("[latex_kb] indexing via:", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
