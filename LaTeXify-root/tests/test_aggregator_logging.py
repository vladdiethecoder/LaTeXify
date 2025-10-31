import sys, json
from pathlib import Path

def test_log_keyword_collision(tmp_path: Path, monkeypatch):
    # Import aggregator and ensure _log accepts details without colliding with its first param
    sys.path.insert(0, str(Path.cwd()))
    from scripts.aggregator import _log, _write_main

    log = tmp_path / "build.log.jsonl"
    _log(log, "unit_test_event", file="x.tex", bytes=1)  # should not raise

    plan = {"tasks":[{"id":"t01","title":"Demo"}], "doc_class":"lix_article"}
    out = tmp_path / "main.tex"
    snips = tmp_path / "snippets"; snips.mkdir()
    (snips / "t01.tex").write_text("\\section*{Task t01}\n\\label{sec:t01}\nOK\n", encoding="utf-8")
    _write_main(out, docclass="lix_article", plan=plan, snippets_dir=snips,
                packages=["geometry"], log_path=log)
    assert out.exists()
