# tests/test_aggregator_logging_collision.py
from __future__ import annotations
from pathlib import Path
import json

from scripts.aggregator import run_aggregator

def test_placeholder_branch_no_collision(tmp_path: Path):
    # Minimal plan with a missing snippet to force the placeholder path
    plan = {
        "doc_class": "lix_article",
        "tasks": [
            {"id": "T01", "order": 1, "anchor": "frontmatter.title", "title": "Title & Metadata"},
            {"id": "T03", "order": 3, "anchor": "introduction", "title": "Introduction"},  # no snippet on disk
        ],
    }
    (tmp_path / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    out_dir = tmp_path / "build"
    res = run_aggregator(tmp_path / "plan.json", tmp_path / "snippets", out_dir, no_compile=True, simulate=True)

    assert (out_dir / "main.tex").exists()
    log = (Path("evidence") / "aggregate.log.jsonl")
    assert log.exists()
    data = log.read_text(encoding="utf-8")
    assert "snippet_missing_placeholder_injected" in data
