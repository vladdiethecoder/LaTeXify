from typing import Dict, Any, List


def mock_layout_plan() -> Dict[str, Any]:
    blocks: List[Dict[str, Any]] = [
        {"id": "c1_text_1", "type": "text", "column": 1, "order": 0, "meta": {"title": "Introduction"}},
        {"id": "c1_code_1", "type": "code", "column": 1, "order": 1, "meta": {"title": "Example Box"}},
        {"id": "c2_math_1", "type": "math", "column": 2, "order": 0, "meta": {"title": "Key Equation"}},
        {"id": "c2_text_2", "type": "text", "column": 2, "order": 1, "meta": {"title": "Notes"}},
    ]
    return {"version": "0.1.0", "columns": 2, "blocks": blocks}


def plan_event() -> Dict[str, Any]:
    return {"type": "plan", "plan": mock_layout_plan()}


__all__ = ["plan_event", "mock_layout_plan"]
