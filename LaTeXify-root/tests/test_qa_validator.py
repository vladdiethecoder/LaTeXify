from __future__ import annotations

import json
from pathlib import Path

import pytest

from latexify.pipeline import qa_validator
from latexify.pipeline import critic_agent


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
    preamble_path = snippets_dir / "PREAMBLE.tex"
    preamble_path.write_text("\\documentclass{article}\n\\begin{document}\n", encoding="utf-8")
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
    assert "\\usepackage{chemfig}" not in report["suggested_packages"]
    finding_codes = {item["code"] for item in report["findings"]}
    assert "auto_flagged" in finding_codes
    assert "environment_imbalance" in finding_codes
    fixes_dir = build_dir / "qa" / "fixes"
    fixed = (fixes_dir / "S1.fixed.tex").read_text(encoding="utf-8")
    assert "\\end{align}" in fixed
    assert "\\usepackage{chemfig}" in preamble_path.read_text(encoding="utf-8")


def test_chktex_detects_math_issues(monkeypatch, tmp_path: Path) -> None:
    plan_path = _write_plan(tmp_path)
    snippets_dir = tmp_path / "snippets"
    snippets_dir.mkdir()
    (snippets_dir / "PREAMBLE.tex").write_text("\\documentclass{article}\n\\begin{document}\n", encoding="utf-8")
    snippet_path = snippets_dir / "S1.tex"
    snippet_path.write_text("$x = y\n", encoding="utf-8")

    fake_chktex = tmp_path / "chktex"
    fake_chktex.write_text(
        """#!/bin/bash
while [[ \"$1\" == -* ]]; do shift; done
echo \"Warning 24 in $1 line 1: Missing $ inserted.\"
exit 1
""",
        encoding="utf-8",
    )
    fake_chktex.chmod(0o755)
    monkeypatch.setattr(qa_validator, "CHK_TEX_BIN", str(fake_chktex))

    report_path = qa_validator.run_preflight(
        plan_path,
        snippets_dir,
        build_dir=tmp_path / "build",
        attempt_compile=False,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    codes = {item["code"] for item in report["findings"]}
    assert "math_delimiter" in codes


def test_compile_failure_triggers_critic(monkeypatch, tmp_path: Path) -> None:
    plan_path = _write_plan(tmp_path)
    snippets_dir = tmp_path / "snippets"
    snippets_dir.mkdir()
    (snippets_dir / "PREAMBLE.tex").write_text("\\documentclass{article}\n\\begin{document}\n", encoding="utf-8")
    (snippets_dir / "S1.tex").write_text("Reference to Eq~\\ref{eq:one}", encoding="utf-8")

    def fake_attempt(main_tex, qa_root):
        log = (
            "LaTeX Warning: Reference `eq:one' on page 1 undefined.\n"
            "! LaTeX Error: File `chemfig.sty' not found.\n"
            "Missing $ inserted.\n"
        )
        summary = qa_validator.CompileSummary(True, False, "pdflatex", None, "latexmk failed")
        return summary, log

    class DummyCritic:
        def __init__(self, plan):
            self.plan = plan

        def review(self, snippet, *, bundle, decision, attempt, feedback_history):
            return critic_agent.ReviewResult(accepted=False, feedback="Snippet failed")

    monkeypatch.setattr(qa_validator, "_attempt_compile", fake_attempt)
    monkeypatch.setattr(qa_validator, "CriticAgent", DummyCritic)

    report_path = qa_validator.run_preflight(
        plan_path,
        snippets_dir,
        build_dir=tmp_path / "build",
        attempt_compile=True,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    codes = {item["code"] for item in report["findings"]}
    assert {"undefined_reference", "missing_package", "math_delimiter", "critic_feedback"}.issubset(codes)
