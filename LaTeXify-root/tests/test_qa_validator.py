from __future__ import annotations

import json
from pathlib import Path

from latexify.pipeline import qa_validator


def _write_plan(tmp_path: Path) -> Path:
    plan = {
        "doc_class": "article",
        "content_flags": {"has_math": True},
        "tasks": [
            {"id": "PREAMBLE", "order": 0},
            {"id": "TITLE", "order": 1},
            {"id": "S1", "order": 2, "title": "Chemistry"},
        ],
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    return path


def test_qa_preflight_reports_and_fixes(tmp_path: Path) -> None:
    plan_path = _write_plan(tmp_path)
    snippets_dir = tmp_path / "snippets"
    snippets_dir.mkdir()
    (snippets_dir / "PREAMBLE.tex").write_text("\\documentclass{article}\n\\begin{document}\n", encoding="utf-8")
    (snippets_dir / "TITLE.tex").write_text("\\maketitle\n", encoding="utf-8")
    snippet_text = r"""\section{Chem}
The molecule is \chemfig{A-B=C}.
\begin{align}
E = mc^2
"""
    snippet_path = snippets_dir / "S1.tex"
    snippet_path.write_text(snippet_text, encoding="utf-8")
    meta = {"auto_flagged": True}
    snippet_path.with_suffix(".meta.json").write_text(json.dumps(meta), encoding="utf-8")
    build_dir = tmp_path / "build"
    report_path = qa_validator.run_preflight(
        plan_path,
        snippets_dir,
        build_dir=build_dir,
        attempt_compile=False,
        max_passes=2,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert "\\usepackage{chemfig}" in report["suggested_packages"]
    finding_codes = {item["code"] for item in report["findings"]}
    assert "auto_flagged" in finding_codes
    assert "environment_imbalance" in finding_codes
    fixes_dir = build_dir / "qa" / "fixes"
    fixed = (fixes_dir / "S1.fixed.tex").read_text(encoding="utf-8")
    assert "\\end{align}" in fixed
