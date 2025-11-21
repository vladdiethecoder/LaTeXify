"""Self-critique helpers for the latexify.pipeline."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List

from ..core import common

TABLE_ENV_RE = re.compile(r"\\begin\{table\}.*?\\end\{table\}", re.DOTALL)
LIST_ENV_RE = re.compile(r"\\begin\{(itemize|enumerate)\}.*?\\end\{\1\}", re.DOTALL)
EQUATION_RE = re.compile(r"\\begin\{equation\}.*?\\end\{equation\}", re.DOTALL)


def evaluate_output(plan_path: Path, tex_path: Path) -> List[str]:
    plan = common.load_plan(plan_path)
    tex = tex_path.read_text(encoding="utf-8")
    issues: List[str] = []
    if not plan:
        issues.append("plan-empty")
    section_blocks = [block for block in plan if block.block_type == "section"]
    if not section_blocks:
        issues.append("missing-sections")
    table_blocks = [b for b in plan if b.block_type == "table"]
    if table_blocks and not TABLE_ENV_RE.search(tex):
        issues.append("table-env-missing")
    list_blocks = [b for b in plan if b.block_type == "list"]
    if list_blocks and not LIST_ENV_RE.search(tex):
        issues.append("list-env-missing")
    equation_blocks = [b for b in plan if b.block_type == "equation"]
    if equation_blocks and len(EQUATION_RE.findall(tex)) < len(equation_blocks):
        issues.append("equation-count-mismatch")
    for block in plan:
        if block.images:
            for image in block.images:
                if Path(image).name not in tex:
                    issues.append(f"image-missing:{Path(image).name}")
                    break
    chunk_ids = {block.chunk_id for block in plan}
    snippets_path = tex_path.with_name("snippets.json")
    snippet_ids = {snippet.chunk_id for snippet in common.load_snippets(snippets_path)} if snippets_path.exists() else set()
    missing = chunk_ids - snippet_ids
    if missing:
        issues.append(f"missing-snippets:{','.join(sorted(missing))}")
    return issues


def attempt_fix(
    plan_path: Path,
    retrieval_path: Path,
    chunks_path: Path,
    snippets_path: Path,
) -> str | None:
    plan = common.load_plan(plan_path)
    changed = False
    if plan and all(block.block_type != "section" for block in plan):
        plan[0].block_type = "section"
        changed = True
    for block in plan:
        region = (block.metadata or {}).get("region_type")
        if region == "table" and block.block_type != "table":
            block.block_type = "table"
            changed = True
        if region == "list" and block.block_type != "list":
            block.block_type = "list"
            changed = True
        if (block.metadata or {}).get("formula_detected") and block.block_type != "equation":
            block.block_type = "equation"
            changed = True
    if changed:
        common.save_plan(plan, plan_path)
        return "aligned-plan-with-structure"
    return None


__all__ = ["evaluate_output", "attempt_fix"]
