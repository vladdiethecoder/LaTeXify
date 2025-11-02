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
  python -m latexify.kb.build_latex_kb
  python -m latexify.kb.build_latex_kb --no_index        # only write JSONL/chunks
  python -m latexify.kb.build_latex_kb --run_dir dev/runs/latex_kb --out indexes/latex_docs.index
"""
from __future__ import annotations

import json
import argparse
from dataclasses import dataclass
from hashlib import sha1
from html import unescape
from pathlib import Path
from typing import Dict, List
import re


@dataclass
class KBRecord:
    """Normalized representation of a lightweight knowledge-base article."""

    id: str
    url: str
    title: str
    question: str
    answer: str
    code_blocks: List[str]
    tags: List[str]


_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(fragment: str) -> str:
    return unescape(_TAG_RE.sub("", fragment or "")).strip()


def _code_blocks(html: str) -> List[str]:
    return [unescape(block.strip()) for block in re.findall(r"<code>(.*?)</code>", html, flags=re.S)]


def _record_id(url: str, title: str) -> str:
    h = sha1()
    h.update(url.encode("utf-8"))
    h.update(title.encode("utf-8"))
    return f"sha1:{h.hexdigest()}"


def parse_texse_html(url: str, html: str) -> KBRecord:
    title_match = re.search(r'<h1[^>]*>(.*?)</h1>', html, flags=re.S)
    question_match = re.search(r'<div[^>]+id="question"[^>]*>(.*?)</div>', html, flags=re.S)
    answer_match = re.search(r'<div[^>]+class="answer[^"]*"[^>]*>.*?<div[^>]+class="js-post-body"[^>]*>(.*?)</div>', html, flags=re.S)
    tags = [unescape(tag.strip()) for tag in re.findall(r'<a[^>]+class="post-tag"[^>]*>(.*?)</a>', html, flags=re.S)]
    title = _strip_html(title_match.group(1) if title_match else url)
    question = _strip_html(question_match.group(1) if question_match else "")
    answer = _strip_html(answer_match.group(1) if answer_match else "")
    return KBRecord(
        id=_record_id(url, title),
        url=url,
        title=title,
        question=question,
        answer=answer,
        code_blocks=_code_blocks(html),
        tags=tags,
    )


def parse_overleaf_html(url: str, html: str) -> KBRecord:
    title_match = re.search(r'<h1[^>]*>(.*?)</h1>', html, flags=re.S)
    paragraph_match = re.search(r'<p>(.*?)</p>', html, flags=re.S)
    article_match = re.search(r'<article[^>]*>(.*?)</article>', html, flags=re.S)
    title = _strip_html(title_match.group(1) if title_match else url)
    question = _strip_html(paragraph_match.group(1) if paragraph_match else "")
    answer = _strip_html(article_match.group(1) if article_match else question)
    page_title = re.search(r'<title>(.*?)</title>', html, flags=re.S)
    tags = [part.strip().lower() for part in (page_title.group(1).split("-") if page_title else []) if part.strip()]
    return KBRecord(
        id=_record_id(url, title),
        url=url,
        title=title,
        question=question,
        answer=answer,
        code_blocks=_code_blocks(html),
        tags=tags,
    )


def parse_wikibooks_html(url: str, html: str) -> KBRecord:
    title_match = re.search(r'<h1[^>]*>(.*?)</h1>', html, flags=re.S)
    content_match = re.search(r'<div[^>]+id="mw-content-text"[^>]*>(.*?)</div>', html, flags=re.S)
    paragraph_match = re.search(r'<p>(.*?)</p>', html, flags=re.S)
    title = _strip_html(title_match.group(1) if title_match else url)
    question = _strip_html(paragraph_match.group(1) if paragraph_match else "")
    answer = _strip_html(content_match.group(1) if content_match else question)
    page_title = re.search(r'<title>(.*?)</title>', html, flags=re.S)
    tags = [part.strip().lower() for part in (page_title.group(1).split("-") if page_title else []) if part.strip()]
    return KBRecord(
        id=_record_id(url, title),
        url=url,
        title=title,
        question=question,
        answer=answer,
        code_blocks=_code_blocks(html),
        tags=tags,
    )

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
            sys.executable,
            "-m",
            "latexify.kb.build_index_bgem3",
            "--run_dir",
            args.run_dir,
            "--out",
            args.out,
        ]
        print("[latex_kb] indexing via:", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
