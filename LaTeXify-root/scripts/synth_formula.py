"""Specialized formula synthesis helper."""

from __future__ import annotations

from typing import Dict, List, Tuple


def synthesize(bundle: Dict) -> Tuple[str, List[str]]:
    formula = (bundle.get("formula_latex") or bundle.get("prompt") or "").strip()
    if not formula:
        formula = "% TODO: formula content unavailable"
    label = bundle.get("id") or "eq"
    body = [
        "\\begin{equation}",
        f"  {formula}",
        f"  \\label{{eq:{label}}}",
        "\\end{equation}",
    ]
    return "\n".join(body) + "\n", ["amsmath"]
