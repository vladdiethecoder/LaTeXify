"""
Tests for the planner scaffold module.

These tests validate that the generated plan enforces unique identifiers and
monotonically increasing order values and that visual assets are routed
correctly when supplied via a layout analysis and asset manifest.  The
imported functions and classes come from ``latexify.pipeline.planner_scaffold``.
"""

from __future__ import annotations

import json
from pathlib import Path

from latexify.pipeline.planner_scaffold import (
    AssetInfo,
    AssetLookup,
    LayoutBlock,
    _emit_plan,
    validate_plan,
)


def test_plan_unique_and_strict_order(tmp_path: Path) -> None:
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


def test_plan_demotes_missing_figure_to_placeholder() -> None:
    blocks = {
        "Q1": LayoutBlock(block_id="Q1", content_type="Figure", page_index=3)
    }
    plan = _emit_plan("lix_article", questions=["Q1"], layout_blocks=blocks)
    task = next(t for t in plan["tasks"] if t["id"] == "Q1")
    assert task["kind"] == "figure_placeholder"
    assert task["type"] == "figure"
    assert "asset_path" not in task
    assert task.get("asset_source_type") == "Figure"
    assert task.get("asset_page_index") == 3


def test_plan_routes_table_image_to_figure() -> None:
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
    assert task["kind"] == "figure"
    assert task["type"] == "table"
    assert task["asset_path"] == "assets/Q2.png"
    assert task.get("asset_source_type") == "Table"
    assert task.get("asset_page_index") == 0
    # Break order
    bad2 = json.loads(json.dumps(plan))
    bad2["tasks"][1]["order"] = bad2["tasks"][0]["order"]
    try:
        validate_plan(bad2)
        assert False, "expected failure on non-increasing order"
    except SystemExit:
        pass


def test_plan_infers_all_content_types() -> None:
    blocks = {
        "QTXT": LayoutBlock(block_id="QTXT", content_type="Paragraph"),
        "QFIG": LayoutBlock(block_id="QFIG", content_type="Figure", page_index=1),
        "QTAB": LayoutBlock(block_id="QTAB", content_type="Table", page_index=2),
        "QMATH": LayoutBlock(block_id="QMATH", content_type="Math", page_index=4),
    }
    lookup = AssetLookup()
    lookup.add(
        AssetInfo(
            asset_path="assets/QFIG.png",
            asset_type="Figure",
            page_index=1,
            block_id="QFIG",
            asset_id="QFIG",
        )
    )
    lookup.add(
        AssetInfo(
            asset_path="assets/QTAB.png",
            asset_type="Table",
            page_index=2,
            block_id="QTAB",
            asset_id="QTAB",
        )
    )
    plan = _emit_plan(
        "lix_article",
        questions=["QTXT", "QFIG", "QTAB", "QMATH"],
        layout_blocks=blocks,
        assets=lookup,
    )
    tasks = {t["id"]: t for t in plan["tasks"]}
    assert tasks["QTXT"].get("type") == "text"
    assert tasks["QTXT"].get("kind") == "content"
    assert tasks["QFIG"].get("type") == "figure"
    assert tasks["QFIG"].get("kind") == "figure"
    assert tasks["QTAB"].get("type") == "table"
    assert tasks["QTAB"].get("kind") == "figure"
    assert tasks["QMATH"].get("type") == "math"
    assert tasks["QMATH"].get("kind") == "content"
    assert "asset_path" not in tasks["QMATH"]
