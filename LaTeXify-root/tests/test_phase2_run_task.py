# tests/test_phase2_run_task.py
from __future__ import annotations
import json
from pathlib import Path

from scripts.phase2_run_task import _load_plan
from scripts.phase2_run_task import main as orch_main  # used via CLI in MSC, here we simulate

def test_orchestrate_writes_snippet(tmp_path: Path, monkeypatch):
    # Minimal plan with T03 present
    plan = {
        "tasks": [
            {"id": "T01", "order": 1, "anchor": "frontmatter.title", "title": "Title & Metadata"},
            {"id": "T03", "order": 3, "anchor": "introduction", "title": "Introduction"},
        ]
    }
    (tmp_path / "plan.json").write_text(json.dumps(plan), encoding="utf-8")

    # Use indices that don't exist to exercise graceful empty retrieval path
    # retrieval_bundle.build_context_bundle already tolerates missing dirs
    snippets = tmp_path / "snippets"
    snippets.mkdir()

    # Run orchestrator as a function by faking argv
    import sys
    old_argv = sys.argv[:]
    sys.argv = [
        "phase2_run_task",
        "--plan", str(tmp_path / "plan.json"),
        "--task_id", "T03",
        "--assignment", str(tmp_path / "missingA"),
        "--assessment", str(tmp_path / "missingB"),
        "--rubric", str(tmp_path / "missingC"),
        "--user", str(tmp_path / "missingD"),
        "--snippets_dir", str(snippets),
    ]
    try:
        from scripts.phase2_run_task import main as _main
        _main()
    finally:
        sys.argv = old_argv

    out_tex = snippets / "T03.tex"
    assert out_tex.exists()
    content = out_tex.read_text(encoding="utf-8")
    assert "\\section{Introduction}" in content
