# latexify/pipeline/model_router.py
from dataclasses import dataclass

@dataclass
class RouteDecision:
    model_name: str
    reason: str = ""

def choose_model_for_text(task: dict) -> RouteDecision:
    """
    Tiny stub. Real version would look at task["kind"] / OCR signal / math density.
    """
    kind = task.get("kind") or task.get("type") or ""
    if "figure" in kind or "image" in kind:
        return RouteDecision(model_name="latexify-figure", reason="figure task")
    if "table" in kind:
        return RouteDecision(model_name="latexify-table", reason="table task")
    return RouteDecision(model_name="latexify-default", reason="fallback")
