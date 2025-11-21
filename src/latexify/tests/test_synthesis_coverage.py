import json
from pathlib import Path

from latexify.core import common
from latexify.pipeline import synthesis_coverage


def test_find_gaps_reports_missing_chunks(tmp_path):
    master_plan = {
        "sections": [
            {
                "section_id": "sec-001",
                "title": "Test Section",
                "content": [
                    {"chunk_id": "c1", "summary": "Intro"},
                    {"chunk_id": "c2", "summary": "Details"},
                ],
            }
        ]
    }
    master_plan_path = tmp_path / "master_plan.json"
    master_plan_path.write_text(json.dumps(master_plan), encoding="utf-8")
    snippets = [common.Snippet(chunk_id="c1", latex="Body", notes={})]
    snippets_path = tmp_path / "snippets.json"
    common.save_snippets(snippets, snippets_path)
    report = synthesis_coverage.find_gaps(master_plan_path, snippets_path)
    assert report["expected_snippets"] == 2
    assert report["actual_snippets"] == 1
    assert report["missing_chunk_ids"] == ["c2"]
