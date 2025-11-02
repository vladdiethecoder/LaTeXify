"""Generate placeholder LaTeX blocks when an expected asset is missing."""

from __future__ import annotations

from typing import Dict, List, Tuple

from .synth_shared import sanitize_inline, slugify

FIGURE_TOKENS = ("figure", "image", "picture", "graphic", "diagram", "photo", "chart")
TABLE_TOKENS = ("table", "spreadsheet", "tabular")
MATH_TOKENS = ("formula", "equation", "math", "expression")


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


def _box_text(environment: str, semantic: str | None, task_id: str | None) -> str:
    if environment == "table":
        base = "Missing Table Image"
    elif semantic and any(token in semantic.lower() for token in MATH_TOKENS):
        base = "Missing Formula Image"
    else:
        base = "Missing Image"
    if task_id:
        return f"{base}: {task_id}"
    return base


def synthesize(bundle: Dict) -> Tuple[str, List[str]]:
    semantic = bundle.get("asset_source_type") or bundle.get("content_type")
    environment, label_prefix = _resolve_environment(semantic)
    caption_raw = bundle.get("caption") or (bundle.get("prompt") or "Missing asset placeholder").splitlines()[0]
    caption = sanitize_inline(caption_raw)
    label = slugify(bundle.get("id") or label_prefix)
    task_id = bundle.get("id")
    box = _box_text(environment, semantic, task_id)

    lines = [
        f"\\begin{{{environment}}}[ht]",
        "  \\centering",
        f"  \\fbox{{{box}}}",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label_prefix}:{label}}}",
        f"\\end{{{environment}}}",
    ]
    return "\n".join(lines) + "\n", []
