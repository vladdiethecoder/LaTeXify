# tests/test_planner_scaffold.py
from __future__ import annotations
import json
from pathlib import Path

<<<<<<< ours
from scripts.planner_scaffold import (
    AssetInfo,
    AssetLookup,
    LayoutBlock,
    _emit_plan,
    validate_plan,
)
=======
from latexify.pipeline.planner_scaffold import _emit_plan, validate_plan
>>>>>>> theirs

def test_plan_unique_and_strict_order(tmp_path: Path):
    plan = _emit_plan("lix_article")
    # Valid baseline
    validate_plan(plan)

    # Break uniqueness
    bad = json.loads(json.dumps(plan))
    bad["tasks"][1]["id"] = bad["tasks"][0]["id"]
    try:
        validate_plan(bad)
        assert False, "expected failure on non-unique IDs"
    except SystemExit:
        pass


def test_plan_demotes_missing_figure_to_placeholder():
    blocks = {"Q1": LayoutBlock(block_id="Q1", content_type="Figure")}
    plan = _emit_plan("lix_article", questions=["Q1"], layout_blocks=blocks)
    task = next(t for t in plan["tasks"] if t["id"] == "Q1")
    assert task["type"] == "figure_placeholder"
    assert "asset_path" not in task
    assert task.get("asset_source_type") == "Figure"


def test_plan_routes_table_image_to_figure():
    blocks = {"Q2": LayoutBlock(block_id="Q2", content_type="Table", page_index=0)}
    lookup = AssetLookup()
    lookup.add(
        AssetInfo(
            asset_path="assets/Q2.png",
            asset_type="Table",
            page_index=0,
            block_id="Q2",
            asset_id="Q2",
        )
    )
    plan = _emit_plan("lix_article", questions=["Q2"], layout_blocks=blocks, assets=lookup)
    task = next(t for t in plan["tasks"] if t["id"] == "Q2")
    assert task["type"] == "figure"
    assert task["asset_path"] == "assets/Q2.png"
    assert task.get("asset_source_type") == "Table"

    # Break order
    bad2 = json.loads(json.dumps(plan))
    bad2["tasks"][1]["order"] = bad2["tasks"][0]["order"]
    try:
        validate_plan(bad2)
        assert False, "expected failure on non-increasing order"
    except SystemExit:
        pass
