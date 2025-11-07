from __future__ import annotations

import json
from pathlib import Path

from latexify.pipeline.layout_planner import LayoutPlanner, LayoutPlannerConfig, main as planner_main


def test_layout_planner_offline_sections(tmp_path):
    cfg = LayoutPlannerConfig(prefer_remote=False)
    planner = LayoutPlanner(cfg)
    draft = (
        "CHAPTER 1: Linear Models\n"
        "This unit introduces slope-intercept form and references Figure 1 for a graph.\n"
        "Table 1 compares growth rates.\n"
        "Section 2 discusses error analysis and includes $$y = mx + b$$."
    )
    blueprint = planner.generate(draft, images=[])
    plan = blueprint.plan
    assert plan["sections"], "expected at least one section"
    assert plan["page_layout"]["columns"] in {"single-column", "two-column"}
    assert blueprint.source["text_chars"] == len(draft)
    assert plan["assets"], "keyword-based visuals should create inferred slots"
    assert plan["content_flags"]["has_figures"]
    assert plan["content_flags"]["has_tables"]
    assert plan["doc_class_hint"]["candidate"]
    assert plan["chunks"], "chunk outline should be present"


def test_layout_planner_cli_offline(tmp_path):
    draft_path = tmp_path / "draft.txt"
    draft_path.write_text("Lesson 1\nKeep findings in a sidebar and show a comparison table.", encoding="utf-8")
    out_json = tmp_path / "layout.json"
    out_txt = tmp_path / "layout.txt"
    exit_code = planner_main(
        [
            "--text-file",
            str(draft_path),
            "--out",
            str(out_json),
            "--text-out",
            str(out_txt),
            "--offline",
        ]
    )
    assert exit_code == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["plan"]["sections"]
    assert payload["plan"]["doc_class_hint"]["candidate"]
    assert payload["plan"]["chunks"]
    assert out_txt.read_text(encoding="utf-8").startswith("Layout blueprint")
