# -*- coding: utf-8 -*-
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"


def write_tex(build: Path, body: str, use_article: bool = True):
    build.mkdir(parents=True, exist_ok=True)
    docclass = "article" if use_article else "lix_textbook"
    (build / "main.tex").write_text(
        "\\documentclass{" + docclass + "}\n"
        "\\usepackage{graphicx}\n"
        "\\begin{document}\n"
        + body +
        "\n\\end{document}\n",
        encoding="utf-8"
    )


def test_trivial_doc_compiles(tmp_path: Path, monkeypatch):
    # Arrange — isolated temp build
    build = tmp_path / "build"
    write_tex(build, "Hello, world.")
    runs_root = tmp_path / "dev" / "runs"
    run_id = "testrun_" + time.strftime("%H%M%S")

    # Act
    rc = subprocess.call([
        sys.executable, str(SCRIPTS / "compile_loop.py"),
        "--main-tex", str(build / "main.tex"),
        "--build-dir", str(build),
        "--runs-root", str(runs_root),
        "--run-id", run_id,
        "--auto-fix", "0",
    ])

    # Assert
    assert rc in (0, 1)  # allow failure if LaTeX missing in CI, but prefer 0
    report = json.loads((runs_root / run_id / "compile" / "compile_report.json").read_text())
    assert "status" in report
    assert report["provenance"] in {"rag", "rule"}
    assert "text_metrics" in report
    assert "promotion_thresholds" in report["text_metrics"]
    assert isinstance(report["promote_to_kb"], bool)
    # If LaTeX is present we expect ok; otherwise fail but report exists
    if shutil.which("pdflatex") or shutil.which("latexmk"):
        assert report["status"] == "ok"


def test_failing_snippet_triggers_auto_fix(tmp_path: Path, monkeypatch):
    build = tmp_path / "build"
    snippets = build / "snippets"
    snippets.mkdir(parents=True, exist_ok=True)
    # Create a broken snippet with an unknown command
    (snippets / "0001.tex").write_text("\\mystery{This should fail}\n", encoding="utf-8")

    body = "Before.\n\\input{snippets/0001.tex}\nAfter."
    write_tex(build, body)
    runs_root = tmp_path / "dev" / "runs"
    run_id = "testrun_" + time.strftime("%H%M%S")

    rc = subprocess.call([
        sys.executable, str(SCRIPTS / "compile_loop.py"),
        "--main-tex", str(build / "main.tex"),
        "--build-dir", str(build),
        "--runs-root", str(runs_root),
        "--run-id", run_id,
        "--auto-fix", "1",
        "--max-retries", "1",
        "--seed", "4242",
    ])

    report = json.loads((runs_root / run_id / "compile" / "compile_report.json").read_text())
    assert "status" in report
    assert report["provenance"] in {"rag", "rule"}
    assert "text_metrics" in report
    assert "promotion_thresholds" in report["text_metrics"]
    assert isinstance(report["promote_to_kb"], bool)
    # Deterministic: we either fix by commenting the unknown line or still fail clearly after one retry
    assert report["passes"] >= 1
    assert report["retry_policy"]["auto_fix_attempted"] is True
    # If LaTeX is available, likely ends 'ok' after the comment fix; otherwise a clean 'fail'
    assert report["status"] in ("ok", "fail")
    # When failing we must include at least one error
    if report["status"] == "fail":
        assert len(report["errors"]) >= 1
