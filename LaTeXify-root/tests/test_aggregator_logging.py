import json
import sys
from pathlib import Path


def test_log_keyword_collision(tmp_path: Path):
    sys.path.insert(0, str(Path.cwd()))
    from scripts.aggregator import _log_event, run_aggregator

    log = tmp_path / "log.jsonl"
    _log_event(log, "unit_test_event", file="x.tex", bytes=1)

    plan = {
        "tasks": [{"id": "t01", "title": "Demo", "order": 1}],
        "doc_class": "lix_article",
    }
    (tmp_path / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    snippets = tmp_path / "snippets"
    snippets.mkdir()
    (snippets / "t01.tex").write_text("\\section*{Task t01}\n\\label{sec:t01}\nOK\n", encoding="utf-8")

    out_dir = tmp_path / "build"
    run_aggregator(tmp_path / "plan.json", snippets, out_dir, no_compile=True, simulate=True)

    main_tex = out_dir / "main.tex"
    assert main_tex.exists()
    main_text = main_tex.read_text(encoding="utf-8")
    assert "\\input{sections/00_demo.tex}" in main_text
