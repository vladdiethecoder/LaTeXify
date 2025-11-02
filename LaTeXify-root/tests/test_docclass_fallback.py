import json
import os
from pathlib import Path
import subprocess
import sys

def test_docclass_fallback_and_no_compile(tmp_path: Path, monkeypatch):
    # Arrange: minimal plan with LiX class
    plan = {
        "doc_class": "lix_article",
        "tasks": [
            {"id": "T03", "order": 3, "anchor": "intro", "title": "Introduction", "depends_on": []}
        ],
    }
    (tmp_path / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    (tmp_path / "snippets").mkdir()
    (tmp_path / "snippets" / "T03.tex").write_text(
        "\\section{Introduction}\n\\label{sec:T03-introduction}\nHello.\n", encoding="utf-8"
    )

    # Force fallback deterministically (no reliance on kpsewhich)
    env = os.environ.copy()
    env["AGG_FORCE_DOCCLASS_FALLBACK"] = "1"

    # Act
    cmd = [
        sys.executable,
        "-m",
        "latexify.pipeline.aggregator",
        "--plan", str(tmp_path / "plan.json"),
        "--snippets_dir", str(tmp_path / "snippets"),
        "--out_dir", str(tmp_path / "build"),
        "--no_compile"]
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path.cwd()), env=env)
    assert out.returncode == 0

    j = json.loads(out.stdout.strip())
    assert j["compile_attempted"] is False
    main_tex = Path(j["main_tex"])
    assert main_tex.exists()

    # Assert the fallback class appears in the documentclass line
    head = main_tex.read_text(encoding="utf-8").splitlines()[0].strip()
    assert head == r"\documentclass{scrartcl}"
