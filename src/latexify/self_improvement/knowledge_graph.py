from __future__ import annotations

from typing import Dict, List, Any, Optional


class KnowledgeGraph:
    """
    Lightweight in-memory graph for tracking relationships between agents, patches, and tasks.
    """

    def __init__(self):
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: List[Dict[str, Any]] = []

    def add_node(self, name: str, data: Optional[Dict[str, Any]] = None) -> None:
        if name not in self.nodes:
            self.nodes[name] = data or {}
        else:
            self.nodes[name].update(data or {})

    def add_edge(self, src: str, relation: str, dst: str, data: Optional[Dict[str, Any]] = None) -> None:
        self.add_node(src, {})
        self.add_node(dst, {})
        self.edges.append({"src": src, "relation": relation, "dst": dst, "data": data or {}})

    def find_edges(self, relation: Optional[str] = None, src: Optional[str] = None, dst: Optional[str] = None) -> List[Dict[str, Any]]:
        results = []
        for e in self.edges:
            if relation and e["relation"] != relation:
                continue
            if src and e["src"] != src:
                continue
            if dst and e["dst"] != dst:
                continue
            results.append(e)
        return results

    def to_dict(self) -> Dict[str, Any]:
        return {"nodes": self.nodes, "edges": self.edges}
