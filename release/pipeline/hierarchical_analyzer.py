"""Detect hierarchical relationships between plan blocks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from ..core import common
from ..core.hierarchical_schema import DocumentHierarchy, HierarchyLevel, HierarchyNode

LEVEL_ORDER = {
    HierarchyLevel.PART: 0,
    HierarchyLevel.CHAPTER: 1,
    HierarchyLevel.SECTION: 2,
    HierarchyLevel.SUBSECTION: 3,
}


@dataclass
class HierarchyAnalysis:
    hierarchy: DocumentHierarchy
    block_parent: Dict[str, str | None] = field(default_factory=dict)
    block_levels: Dict[str, HierarchyLevel | None] = field(default_factory=dict)
    block_paths: Dict[str, List[str]] = field(default_factory=dict)
    section_nodes: Dict[str, str] = field(default_factory=dict)

    def parent_for(self, block_id: str) -> str | None:
        return self.block_parent.get(block_id)

    def level_for(self, block_id: str) -> HierarchyLevel | None:
        return self.block_levels.get(block_id)

    def path_for(self, block_id: str) -> List[str]:
        return self.block_paths.get(block_id, [])


def analyze_plan(
    plan: List[common.PlanBlock],
    chunk_map: Dict[str, common.Chunk] | None = None,
) -> HierarchyAnalysis:
    """Assigns part/chapter/section/subsection structure to plan blocks."""
    analysis = HierarchyAnalysis(hierarchy=DocumentHierarchy())
    stack: List[HierarchyNode] = []
    chunk_map = chunk_map or {}

    for block in plan:
        chunk = chunk_map.get(block.chunk_id)
        metadata = _metadata_for(block, chunk)
        level = _infer_level(block, metadata)
        if level:
            _prune_stack(stack, level)
            parent_node = stack[-1] if stack else analysis.hierarchy.root
            node = analysis.hierarchy.register_node(
                level,
                block.label or level.value.title(),
                block_id=block.block_id,
                chunk_id=block.chunk_id,
                page=chunk.page if chunk else None,
                metadata={
                    "header_level": metadata.get("header_level"),
                    "font_size": metadata.get("font_size"),
                    "column": metadata.get("column"),
                },
                parent_id=parent_node.node_id,
            )
            stack.append(node)
            analysis.block_levels[block.block_id] = level
            analysis.section_nodes[block.block_id] = node.node_id
            parent_block_id = parent_node.block_id if parent_node.block_id else None
            analysis.block_parent[block.block_id] = parent_block_id
        else:
            parent_node = stack[-1] if stack else analysis.hierarchy.root
            parent_block_id = parent_node.block_id if parent_node.block_id else None
            analysis.block_parent[block.block_id] = parent_block_id
            analysis.block_levels[block.block_id] = None

        path = [node.title for node in stack]
        analysis.block_paths[block.block_id] = path
        _apply_metadata(block, level, path, analysis.block_parent.get(block.block_id))
    return analysis


def _prune_stack(stack: List[HierarchyNode], incoming_level: HierarchyLevel) -> None:
    incoming_order = LEVEL_ORDER[incoming_level]
    while stack:
        top = stack[-1]
        if LEVEL_ORDER.get(top.level, incoming_order) >= incoming_order:
            stack.pop()
        else:
            break


def _metadata_for(block: common.PlanBlock, chunk: common.Chunk | None) -> Dict[str, object]:
    metadata: Dict[str, object] = {}
    if chunk and chunk.metadata:
        metadata.update(chunk.metadata)
    if block.metadata:
        for key, value in block.metadata.items():
            metadata.setdefault(key, value)
    return metadata


def _infer_level(block: common.PlanBlock, metadata: Dict[str, object]) -> HierarchyLevel | None:
    label = (block.label or "").strip().lower()
    header_level = int(metadata.get("header_level") or 0)
    font_size = float(metadata.get("font_size") or 0.0)
    column = metadata.get("column", 1)
    region = metadata.get("region_type")
    is_heading = block.block_type == "section" or region == "heading" or header_level > 0
    if not is_heading:
        return None
    if "part " in label or label.startswith("part ") or (font_size >= 20 and header_level <= 1):
        return HierarchyLevel.PART
    if label.startswith("chapter") or label.startswith("appendix") or header_level == 1:
        return HierarchyLevel.CHAPTER
    if header_level == 2 or (font_size >= 15 and column == 1):
        return HierarchyLevel.SECTION
    if header_level >= 3:
        return HierarchyLevel.SUBSECTION
    if block.block_type == "section":
        return HierarchyLevel.SECTION if header_level <= 2 else HierarchyLevel.SUBSECTION
    return None


def _apply_metadata(
    block: common.PlanBlock,
    level: HierarchyLevel | None,
    path: List[str],
    parent_block: str | None,
) -> None:
    metadata = block.metadata
    if level:
        metadata["hierarchy_level"] = level.value
    elif "hierarchy_level" in metadata:
        metadata.pop("hierarchy_level", None)
    metadata["hierarchy_path"] = path
    if parent_block:
        metadata["hierarchy_parent"] = parent_block
    else:
        metadata.pop("hierarchy_parent", None)


__all__ = ["HierarchyAnalysis", "analyze_plan"]
