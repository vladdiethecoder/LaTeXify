# tests/test_aggregator.py
from __future__ import annotations
from pathlib import Path
import json

from scripts.aggregator import run_aggregator

def test_aggregator_writes_main(tmp_path: Path):
    # Plan with two tasks
    plan = {
        "doc_class": "lix_article",
        "tasks": [
            {"id": "T01", "order": 1, "anchor": "frontmatter.title", "title": "Title & Metadata"},
            {"id": "T03", "order": 3, "anchor": "introduction", "title": "Introduction"},
        ],
    }
    (tmp_path / "plan.json").write_text(json.dumps(plan), encoding="utf-8")

    snip_dir = tmp_path / "snippets"
    snip_dir.mkdir()
    # T01 frontmatter defines \title but no \maketitle
    (snip_dir / "T01.tex").write_text("\\title{Demo Title}\n\\author{You}\n\\date{}", encoding="utf-8")
    # T03
    (snip_dir / "T03.tex").write_text("\\section{Introduction}\n\\label{sec:T03-introduction}\nHello.", encoding="utf-8")

    out_dir = tmp_path / "build"
    res = run_aggregator(tmp_path / "plan.json", snip_dir, out_dir, no_compile=True, simulate=True)

    main_tex = out_dir / "main.tex"
    assert main_tex.exists()
    tex = main_tex.read_text(encoding="utf-8")
    assert f"\\documentclass{{{res['doc_class']}}}" in tex
    assert "\\input{preamble.tex}" in tex
    assert "\\maketitle" in tex

    # Ensure frontmatter input appears before \maketitle
    first_input = tex.split("\\maketitle", 1)[0]
    assert "sections/00_title_metadata.tex" in first_input

    sections_dir = out_dir / "sections"
    assert sections_dir.exists()
    section_files = sorted(p.name for p in sections_dir.glob("*.tex"))
    assert "00_title_metadata.tex" in section_files
    assert "01_introduction.tex" in section_files

    frontmatter_text = (sections_dir / "00_title_metadata.tex").read_text(encoding="utf-8")
    assert "\\title{Demo Title}" in frontmatter_text
    assert "\\author{You}" in frontmatter_text

    intro_text = (sections_dir / "01_introduction.tex").read_text(encoding="utf-8")
    assert "\\label{sec:T03-introduction}" in intro_text
    assert res["compile_attempted"] is False
