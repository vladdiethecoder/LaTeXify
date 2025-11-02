"""Specialized figure synthesis helper."""

from __future__ import annotations

from typing import Dict, List, Tuple

FIGURE_TOKENS = ("figure", "image", "picture", "graphic", "diagram", "photo", "chart")
TABLE_TOKENS = ("table", "spreadsheet", "tabular")
MATH_TOKENS = ("formula", "equation", "math", "expression")


def _sanitize_label(raw: str) -> str:
    raw = (raw or "figure").lower()
    cleaned = [
        ch if ch.isalnum() or ch in {"-", "_"} else "-"
        for ch in raw
    ]
    label = "".join(cleaned).strip("-")
    return label or "figure"


def _resolve_environment(kind: str | None) -> Tuple[str, str]:
    if not kind:
        return "figure", "fig"
    lower = kind.lower()
    if any(token in lower for token in TABLE_TOKENS):
        return "table", "tab"
    if any(token in lower for token in FIGURE_TOKENS):
        return "figure", "fig"
    if any(token in lower for token in MATH_TOKENS):
        return "figure", "fig"
    return "figure", "fig"


def synthesize(bundle: Dict) -> Tuple[str, List[str]]:
<<<<<<< ours
    image_path = bundle.get("asset_path") or bundle.get("image_path") or "figure-placeholder.pdf"
=======
    asset_path = bundle.get("asset_path") or bundle.get("image_path")
    if not asset_path:
        raise ValueError("Figure bundle missing 'asset_path'. Provide an asset_path in the plan/bundle.")

    semantic = bundle.get("asset_source_type") or bundle.get("figure_type") or bundle.get("content_type")
    environment, label_prefix = _resolve_environment(semantic)
>>>>>>> theirs
    caption = bundle.get("caption") or (bundle.get("prompt") or "Auto-generated figure").splitlines()[0]
    label = _sanitize_label(bundle.get("id") or label_prefix)

    lines = [
        f"\\begin{{{environment}}}[ht]",
        "  \\centering",
        f"  \\includegraphics[width=\\linewidth]{{{asset_path}}}",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label_prefix}:{label}}}",
        f"\\end{{{environment}}}",
    ]
    return "\n".join(lines) + "\n", ["graphicx"]
