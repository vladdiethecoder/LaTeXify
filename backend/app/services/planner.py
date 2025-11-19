from typing import Any, Dict


def plan_event(plan_payload: Dict[str, Any]) -> Dict[str, Any]:
    return {"type": "plan", "plan": plan_payload}


__all__ = ["plan_event"]
