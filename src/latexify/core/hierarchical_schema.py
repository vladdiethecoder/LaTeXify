"""Shared data structures for hierarchical document analysis and references.

plan.json schema:
    [
        {
            "block_id": str,
            "chunk_id": str,
            "label": str,
            "block_type": str,
            "images": [str],
            "metadata": {
                "region_type": str,
                "header_level": int,
                "...": any extension keys (list_depth, formula_detected, etc.)
            }
        },
        ...
    ]

graph.json schema:
    {
        "nodes": [
            {
                "node_id": str,
                "type": str,
                "label": str,
                "metadata": dict (chunk_id, page, hierarchy metadata, etc.),
                "children": [str],
                "parent": str | None
            }
        ],
        "edges": [
            {"source": str, "target": str, "type": "order" | "hierarchy"}
        ],
        "reading_order": [block_id, ...]
    }

Extension points for hierarchical information live inside the per-block metadata in
plan.json (`hierarchy_level`, `hierarchy_path`, `hierarchy_parent`) and inside graph
nodes via the `metadata["hierarchy"]` payload. The dataclasses below track part/
chapter/section/subsection nodes along with a reference index that keeps figures,
tables, and equations consistently labeled across the pipeline.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Literal, Optional


class HierarchyLevel(str, Enum):
    DOCUMENT = "document"
    PART = "part"
    CHAPTER = "chapter"
    SECTION = "section"
    SUBSECTION = "subsection"


@dataclass
class HierarchyNode:
    node_id: str
    title: str
    level: HierarchyLevel
    block_id: str | None = None
    chunk_id: str | None = None
    page: int | None = None
    metadata: Dict[str, object] = field(default_factory=dict)
    parent_id: str | None = None
    children: List[str] = field(default_factory=list)


@dataclass
class DocumentNode(HierarchyNode):
    def __init__(self, node_id: str, title: str = "Document") -> None:
        super().__init__(node_id=node_id, title=title, level=HierarchyLevel.DOCUMENT)


@dataclass
class PartNode(HierarchyNode):
    def __init__(self, node_id: str, title: str, **kwargs) -> None:
        super().__init__(node_id=node_id, title=title, level=HierarchyLevel.PART, **kwargs)


@dataclass
class ChapterNode(HierarchyNode):
    def __init__(self, node_id: str, title: str, **kwargs) -> None:
        super().__init__(node_id=node_id, title=title, level=HierarchyLevel.CHAPTER, **kwargs)


@dataclass
class SectionNode(HierarchyNode):
    def __init__(self, node_id: str, title: str, **kwargs) -> None:
        super().__init__(node_id=node_id, title=title, level=HierarchyLevel.SECTION, **kwargs)


@dataclass
class SubsectionNode(HierarchyNode):
    def __init__(self, node_id: str, title: str, **kwargs) -> None:
        super().__init__(node_id=node_id, title=title, level=HierarchyLevel.SUBSECTION, **kwargs)


NODE_FACTORY: Dict[HierarchyLevel, type[HierarchyNode]] = {
    HierarchyLevel.DOCUMENT: DocumentNode,
    HierarchyLevel.PART: PartNode,
    HierarchyLevel.CHAPTER: ChapterNode,
    HierarchyLevel.SECTION: SectionNode,
    HierarchyLevel.SUBSECTION: SubsectionNode,
}


class DocumentHierarchy:
    """Tracks the section tree and exposes helpers for resolving context."""

    def __init__(self, root_id: str = "doc-root") -> None:
        self.root_id = root_id
        self.nodes: Dict[str, HierarchyNode] = {}
        self.block_index: Dict[str, str] = {}
        self._counters: Dict[HierarchyLevel, int] = defaultdict(int)
        self._ensure_root()

    def _ensure_root(self) -> None:
        if self.root_id not in self.nodes:
            self.nodes[self.root_id] = DocumentNode(node_id=self.root_id)

    @property
    def root(self) -> HierarchyNode:
        return self.nodes[self.root_id]

    def _next_id(self, level: HierarchyLevel) -> str:
        self._counters[level] += 1
        return f"{level.value}-{self._counters[level]:03d}"

    def register_node(
        self,
        level: HierarchyLevel,
        title: str,
        *,
        block_id: str | None = None,
        chunk_id: str | None = None,
        page: int | None = None,
        metadata: Dict[str, object] | None = None,
        parent_id: str | None = None,
        node_id: str | None = None,
    ) -> HierarchyNode:
        node_key = node_id or self._next_id(level)
        node_cls = NODE_FACTORY[level]
        node = node_cls(
            node_id=node_key,
            title=title or level.value.title(),
            block_id=block_id,
            chunk_id=chunk_id,
            page=page,
            metadata=metadata or {},
        )
        self.nodes[node_key] = node
        if block_id:
            self.block_index[block_id] = node_key
        self.link(parent_id or self.root_id, node_key)
        return node

    def link(self, parent_id: str, child_id: str) -> None:
        parent = self.nodes.get(parent_id)
        child = self.nodes.get(child_id)
        if not parent or not child:
            return
        child.parent_id = parent.node_id
        if child_id not in parent.children:
            parent.children.append(child_id)

    def node_for_block(self, block_id: str) -> HierarchyNode | None:
        node_id = self.block_index.get(block_id)
        return self.nodes.get(node_id) if node_id else None

    def path_to(self, node_id: str) -> List[str]:
        path: List[str] = []
        current = self.nodes.get(node_id)
        while current and current.node_id != self.root_id:
            path.append(current.title)
            if not current.parent_id:
                break
            current = self.nodes.get(current.parent_id)
        return list(reversed(path))


@dataclass
class ReferenceLabel:
    name: str
    target_type: str
    block_id: str | None = None
    chunk_id: str | None = None
    generated: bool = False
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class CrossReference:
    source_block_id: str
    command: str
    target: str
    resolved_label: str | None
    status: Literal["resolved", "unresolved"]
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class ReferenceIndex:
    labels: Dict[str, ReferenceLabel] = field(default_factory=dict)
    by_block: Dict[str, ReferenceLabel] = field(default_factory=dict)
    by_chunk: Dict[str, ReferenceLabel] = field(default_factory=dict)
    references: List[CrossReference] = field(default_factory=list)
    _usage: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def register_label(
        self,
        name: str,
        block_id: str | None,
        target_type: str,
        *,
        chunk_id: str | None = None,
        generated: bool = False,
        metadata: Dict[str, object] | None = None,
    ) -> ReferenceLabel:
        candidate = name.strip()
        if not candidate:
            candidate = target_type
        if candidate in self.labels and self.labels[candidate].block_id != block_id:
            candidate = self._make_unique(candidate)
        label = ReferenceLabel(
            name=candidate,
            target_type=target_type,
            block_id=block_id,
            chunk_id=chunk_id,
            generated=generated,
            metadata=metadata or {},
        )
        self.labels[candidate] = label
        if block_id and block_id not in self.by_block:
            self.by_block[block_id] = label
        if chunk_id and chunk_id not in self.by_chunk:
            self.by_chunk[chunk_id] = label
        return label

    def _make_unique(self, name: str) -> str:
        self._usage[name] += 1
        return f"{name}-{self._usage[name]}"

    def ensure_label(
        self,
        preferred_name: str,
        block_id: str,
        target_type: str,
        *,
        chunk_id: str | None = None,
    ) -> ReferenceLabel:
        existing = self.by_block.get(block_id)
        if existing:
            return existing
        return self.register_label(
            preferred_name,
            block_id,
            target_type,
            chunk_id=chunk_id,
            generated=True,
        )

    def label_for_block(self, block_id: str) -> Optional[str]:
        label = self.by_block.get(block_id)
        return label.name if label else None

    def label_for_chunk(self, chunk_id: str) -> Optional[str]:
        label = self.by_chunk.get(chunk_id)
        return label.name if label else None

    def link_reference(self, command: str, target: str, source_block_id: str) -> CrossReference:
        label = self.labels.get(target) or self.by_block.get(target) or self.by_chunk.get(target)
        resolved = label.name if label else None
        status: Literal["resolved", "unresolved"] = "resolved" if resolved else "unresolved"
        cross = CrossReference(
            source_block_id=source_block_id,
            command=command,
            target=target,
            resolved_label=resolved,
            status=status,
            metadata={"generated": getattr(label, "generated", False) if label else False},
        )
        self.references.append(cross)
        return cross


__all__ = [
    "HierarchyLevel",
    "HierarchyNode",
    "DocumentNode",
    "PartNode",
    "ChapterNode",
    "SectionNode",
    "SubsectionNode",
    "DocumentHierarchy",
    "ReferenceLabel",
    "CrossReference",
    "ReferenceIndex",
]
