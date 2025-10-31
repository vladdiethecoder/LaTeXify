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

    assert (out_dir / "main.tex").exists()
    tex = (out_dir / "main.tex").read_text(encoding="utf-8")
    assert "\\documentclass{lix_article}" in tex
    assert "\\usepackage{amsmath}" in tex
    assert "\\maketitle" in tex
    # Dedup label should not trigger here, only one label present
    assert "\\label{sec:T03-introduction}" in tex
    assert res["compile_attempted"] is False
