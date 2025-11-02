# -*- coding: utf-8 -*-
import json
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"


def test_auto_fix_comment_unknown_command(tmp_path: Path):
    build = tmp_path / "build"
    build.mkdir(parents=True, exist_ok=True)
    f = build / "snippets" / "bad.tex"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("\\mystery{oops}\n", encoding="utf-8")

    run_dir = tmp_path / "dev" / "runs" / ("testrun_" + time.strftime("%H%M%S")) / "compile"
    run_dir.mkdir(parents=True, exist_ok=True)

    proc = subprocess.run(
        [
            sys.executable, str(SCRIPTS / "auto_fix.py"),
            "--error-category", "undefined_control_sequence",
            "--message", "! Undefined control sequence.",
            "--project-root", str(REPO),
            "--build-dir", str(build),
            "--run-dir", str(run_dir),
            "--file", str(f),
            "--line", "1",
            "--seed", "123",
        ],
        capture_output=True,
        text=True,
    )
    out = json.loads(proc.stdout.strip())
    assert out["status"] in ("fixed", "skipped")
    if out["status"] == "fixed":
        assert out["changed_file"].endswith("bad.tex")


def test_auto_fix_missing_asset_placeholder(tmp_path: Path):
    build = tmp_path / "build"
    snippet = build / "snippets" / "figure.tex"
    snippet.parent.mkdir(parents=True, exist_ok=True)
    snippet.write_text(
        "\\begin{figure}[ht]\n  \\includegraphics{assets/foo.png}\n\\end{figure}\n",
        encoding="utf-8",
    )

    run_dir = tmp_path / "dev" / "runs" / ("testrun_" + time.strftime("%H%M%S")) / "compile"
    run_dir.mkdir(parents=True, exist_ok=True)

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "auto_fix.py"),
            "--error-category",
            "file_not_found",
            "--message",
            "LaTeX Error: File `assets/foo.png` not found.",
            "--project-root",
            str(REPO),
            "--build-dir",
            str(build),
            "--run-dir",
            str(run_dir),
            "--file",
            str(snippet),
            "--snippet",
            str(snippet),
            "--seed",
            "123",
        ],
        capture_output=True,
        text=True,
    )
    out = json.loads(proc.stdout.strip())
    assert out["status"] == "fixed"
    assert out["changed_file"].endswith("figure.tex")
    rewritten = snippet.read_text(encoding="utf-8")
    assert "Missing Asset: foo.png" in rewritten
