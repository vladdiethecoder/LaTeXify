"""Cross-reference resolution utilities for LaTeX assembly."""

from __future__ import annotations

import re
from typing import Dict, Tuple

from ..core import common
from ..core.hierarchical_schema import ReferenceIndex

LABEL_RE = re.compile(r"\\label\{([^}]+)\}")
REF_RE = re.compile(r"\\(ref|eqref|autoref)\{([^}]+)\}")
EQUATION_END_RE = re.compile(r"\\end\{equation\*?\}")
FIGURE_END_RE = re.compile(r"\\end\{figure\}")
TABLE_END_RE = re.compile(r"\\end\{table\}")
CAPTION_RE = re.compile(r"\\caption\{[^}]*\}")
AUTO_LABEL_PREFIX = {"figure": "fig", "table": "tbl", "equation": "eq"}


def resolve_references(
    plan: Dict[str, common.PlanBlock] | list[common.PlanBlock],
    snippets: Dict[str, str],
) -> Tuple[Dict[str, str], ReferenceIndex]:
    """Injects missing labels and rewrites \\ref commands."""
    updated = dict(snippets)
    plan_blocks = list(plan)
    index = ReferenceIndex()
    _collect_existing_labels(plan_blocks, updated, index)
    _ensure_auto_labels(plan_blocks, updated, index)
    for block in plan_blocks:
        snippet = updated.get(block.chunk_id, "")
        if not snippet:
            continue
        updated_snippet = _rewrite_references(snippet, block.block_id, index)
        updated[block.chunk_id] = updated_snippet
    return updated, index


def _collect_existing_labels(
    plan: list[common.PlanBlock], snippets: Dict[str, str], index: ReferenceIndex
) -> None:
    for block in plan:
        snippet = snippets.get(block.chunk_id, "")
        if not snippet:
            continue
        for name in LABEL_RE.findall(snippet):
            index.register_label(name, block.block_id, block.block_type, chunk_id=block.chunk_id)


def _ensure_auto_labels(
    plan: list[common.PlanBlock],
    snippets: Dict[str, str],
    index: ReferenceIndex,
) -> None:
    used_names = set(index.labels.keys())
    for block in plan:
        prefix = AUTO_LABEL_PREFIX.get(block.block_type)
        if not prefix:
            continue
        label_hint = index.by_block.get(block.block_id)
        if label_hint:
            block.metadata.setdefault("resolved_label", label_hint.name)
            continue
        slug = _slugify(block.label or block.block_id)
        candidate = f"{prefix}:{slug}"
        if candidate in used_names:
            candidate = f"{candidate}-{len(used_names)}"
        label = index.ensure_label(candidate, block.block_id, block.block_type, chunk_id=block.chunk_id)
        used_names.add(label.name)
        block.metadata.setdefault("labels", []).append(label.name)
        block.metadata["resolved_label"] = label.name
        if block.block_type != "figure":
            snippets[block.chunk_id] = _inject_label(snippets.get(block.chunk_id, ""), label.name, block.block_type)


def _inject_label(snippet: str, label: str, block_type: str) -> str:
    if not snippet.strip():
        return snippet
    if LABEL_RE.search(snippet):
        return snippet
    if block_type == "equation":
        match = EQUATION_END_RE.search(snippet)
        if match:
            insert_at = match.start()
            return snippet[:insert_at] + f"\\label{{{label}}}\n" + snippet[insert_at:]
    if block_type in {"figure", "table"}:
        caption = CAPTION_RE.search(snippet)
        if caption:
            insert_at = caption.end()
            return snippet[:insert_at] + f"\n  \\label{{{label}}}" + snippet[insert_at:]
        terminal = (FIGURE_END_RE if block_type == "figure" else TABLE_END_RE).search(snippet)
        if terminal:
            insert_at = terminal.start()
            return snippet[:insert_at] + f"  \\label{{{label}}}\n" + snippet[insert_at:]
    return snippet + f"\n\\label{{{label}}}"


def _rewrite_references(snippet: str, block_id: str, index: ReferenceIndex) -> str:
    def _replacement(match: re.Match[str]) -> str:
        command, target = match.group(1), match.group(2)
        cross = index.link_reference(command, target, block_id)
        if cross.resolved_label:
            return f"\\{command}{{{cross.resolved_label}}}"
        return match.group(0)

    return REF_RE.sub(_replacement, snippet)


def _slugify(text: str) -> str:
    normalized = "".join(char.lower() if char.isalnum() else "-" for char in text)
    normalized = normalized.strip("-")
    normalized = re.sub("-{2,}", "-", normalized)
    return normalized or "item"


__all__ = ["resolve_references"]
