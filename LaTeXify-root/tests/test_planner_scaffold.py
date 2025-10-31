# tests/test_planner_scaffold.py
from __future__ import annotations
import json
from pathlib import Path

from scripts.planner_scaffold import validate_plan, _emit_plan

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

    # Break order
    bad2 = json.loads(json.dumps(plan))
    bad2["tasks"][1]["order"] = bad2["tasks"][0]["order"]
    try:
        validate_plan(bad2)
        assert False, "expected failure on non-increasing order"
    except SystemExit:
        pass
