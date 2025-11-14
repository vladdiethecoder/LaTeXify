"""Document graph builder inspired by Detect-Order-Construct."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from ..core import common


@dataclass
class GraphNode:
    node_id: str
    type: str
    label: str
    metadata: Dict[str, object] = field(default_factory=dict)
    children: List[str] = field(default_factory=list)


def build_graph(plan_path: Path, chunks_path: Path, output_path: Path) -> Path:
    plan = common.load_plan(plan_path)
    chunks = {chunk.chunk_id: chunk for chunk in common.load_chunks(chunks_path)}
    nodes: Dict[str, GraphNode] = {}
    edges: List[Dict[str, str]] = []
    section_stack: List[GraphNode] = []
    root = GraphNode(node_id="root", type="document", label="Document", metadata={})
    nodes[root.node_id] = root

    for block in plan:
        chunk = chunks.get(block.chunk_id)
        metadata = block.metadata or {}
        header_level = metadata.get("header_level", 0)
        node_type = block.block_type
        node = GraphNode(
            node_id=block.block_id,
            type=node_type,
            label=block.label,
            metadata={
                "chunk_id": block.chunk_id,
                "page": chunk.page if chunk else None,
                "region_type": metadata.get("region_type"),
                "images": block.images,
            },
        )
        nodes[node.node_id] = node
        parent = root
        if node_type == "section":
            while section_stack and section_stack[-1].metadata.get("header_level", 1) >= header_level:
                section_stack.pop()
            parent = section_stack[-1] if section_stack else root
            node.metadata["header_level"] = header_level or 1
            section_stack.append(node)
        else:
            parent = section_stack[-1] if section_stack else root
        parent.children.append(node.node_id)
        edges.append({"source": parent.node_id, "target": node.node_id, "type": "order"})
        if chunk and chunk.metadata.get("image_refs"):
            for image in chunk.metadata["image_refs"]:
                image_node_id = f"{node.node_id}_img_{Path(image).name}"
                image_node = GraphNode(
                    node_id=image_node_id,
                    type="image",
                    label=Path(image).name,
                    metadata={"source_path": image, "parent": node.node_id},
                )
                nodes[image_node_id] = image_node
                node.children.append(image_node_id)
                edges.append({"source": node.node_id, "target": image_node_id, "type": "figure-ref"})
    graph_payload = {
        "nodes": [node.__dict__ for node in nodes.values()],
        "edges": edges,
        "reading_order": [block.block_id for block in plan],
    }
    output_path.write_text(json.dumps(graph_payload, indent=2), encoding="utf-8")
    return output_path


__all__ = ["build_graph"]
