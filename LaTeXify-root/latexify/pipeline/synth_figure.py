"""Specialised figure synthesis helper."""

from __future__ import annotations

from typing import Dict, List, Tuple

from .synth_shared import sanitize_inline, slugify

# Tokens used to infer environment types
FIGURE_TOKENS = ("figure", "image", "picture", "graphic", "diagram", "photo", "chart")
TABLE_TOKENS = ("table", "spreadsheet", "tabular")
MATH_TOKENS = ("formula", "equation", "math", "expression")


def _resolve_environment(kind: str | None) -> Tuple[str, str]:
    """Map a semantic kind to a LaTeX environment and label prefix."""
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
    """Generate a LaTeX figure/table snippet from a bundle.

    The bundle may include an explicit ``asset_path`` or ``image_path``.  If
    neither is provided a placeholder path is used to ensure compilation still
    succeeds.  Semantic hints (``asset_source_type``, ``figure_type``,
    ``content_type``) are used to decide whether to emit a ``figure`` or
    ``table`` environment and to choose the appropriate label prefix.  The
    caption is taken from ``caption`` or the first line of ``prompt`` or
    defaults to a generic description.  Labels are automatically sanitised.
    """
    # Determine the path to include; fall back to a placeholder to avoid errors
    asset_path = bundle.get("asset_path") or bundle.get("image_path")
    if not asset_path:
        asset_path = "figure-placeholder.pdf"
    # Resolve environment and label prefix based on semantic hints
    semantic = (
        bundle.get("asset_source_type")
        or bundle.get("figure_type")
        or bundle.get("content_type")
    )
    environment, label_prefix = _resolve_environment(semantic)
    # Caption: explicit caption > first line of prompt > default
    caption_raw = bundle.get("caption") or (bundle.get("prompt") or "Auto-generated figure").splitlines()[0]
    caption = sanitize_inline(caption_raw)
    # Sanitise the label using id or prefix
    label = slugify(bundle.get("id") or label_prefix)
    # Build snippet lines
    lines = [
        f"\\begin{{{environment}}}[ht]",
        "  \\centering",
        f"  \\includegraphics[width=\\linewidth]{{{asset_path}}}",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label_prefix}:{label}}}",
        f"\\end{{{environment}}}",
    ]
    # Return the snippet and required packages
    return "\n".join(lines) + "\n", ["graphicx"]