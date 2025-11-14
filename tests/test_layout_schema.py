import json
from pathlib import Path

import fastjsonschema

from backend.app.services.planner import mock_layout_plan


SCHEMA_PATH = Path("schemas/layout_plan.schema.json")


def test_layout_plan_schema_validates():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    validate = fastjsonschema.compile(schema)
    plan = mock_layout_plan()
    validate(plan)
    assert plan["columns"] == 2
    assert len(plan["blocks"]) >= 1
