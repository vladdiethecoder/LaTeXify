"""Enhanced hierarchy+semantic graph utilities derived from planner outputs."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple, TYPE_CHECKING

from ..core import common

if TYPE_CHECKING:  # pragma: no cover - typing aid
    from .planner import MasterPlan
else:  # pragma: no cover - runtime placeholder
    MasterPlan = Any  # type: ignore

STOPWORDS = {
    "the",
    "and",
    "or",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "are",
    "was",
    "were",
    "such",
    "using",
    "between",
    "their",
    "have",
}
REFERENCE_RE = re.compile(r"(figure|table|section|eqn)\s+(\d+)", re.IGNORECASE)


class HierarchicalDocumentGraph:
    """Constructs a multi-file document graph with semantic overlays."""

    def __init__(self, plan: MasterPlan, chunk_map: Dict[str, common.Chunk]) -> None:
        self.plan = plan
        self.chunk_map = chunk_map
        self._ordered_chunk_ids = self._ordered_chunks()
        self._ordinal_lookup = self._build_ordinals()

    def build_graph(self) -> Dict[str, object]:
        nodes: List[Dict[str, object]] = []
        edges: List[Dict[str, object]] = []
        covered_chunks = set()
        root_id = "document_root"
        nodes.append({"id": root_id, "type": "document", "label": self.plan.document_title})
        section_nodes: List[str] = []
        content_nodes: List[str] = []
        for idx, section in enumerate(self.plan.sections, start=1):
            node_id = f"section_{idx:03d}"
            heading_chunk = self.chunk_map.get(section.heading_chunk_id or "")
            metadata = {
                "header_level": section.header_level,
                "chunk_id": section.heading_chunk_id,
                "page": heading_chunk.page if heading_chunk else None,
            }
            nodes.append({"id": node_id, "type": "section", "label": section.title, "metadata": metadata})
            section_nodes.append(node_id)
            edges.append({"source": root_id, "target": node_id, "type": "contains"})
            if heading_chunk:
                covered_chunks.add(heading_chunk.chunk_id)
            for content in section.content:
                content_chunk = self.chunk_map.get(content.chunk_id)
                if not content_chunk:
                    continue
                covered_chunks.add(content_chunk.chunk_id)
                content_id = f"content_{content.chunk_id}"
                metadata = {
                    "chunk_id": content.chunk_id,
                    "region_type": content_chunk.metadata.get("region_type"),
                    "page": content_chunk.page,
                    "layout_confidence": content_chunk.metadata.get("layout_confidence"),
                }
                nodes.append({"id": content_id, "type": "chunk", "label": (content.summary or ""), "metadata": metadata})
                edges.append({"source": node_id, "target": content_id, "type": "orders"})
                content_nodes.append(content_id)
        metrics = self._validation_metrics(len(nodes), len(edges), len(section_nodes), len(content_nodes), len(covered_chunks))
        return {"nodes": nodes, "edges": edges, "metrics": metrics}

    def cross_reference_map(self) -> Dict[str, List[Dict[str, object]]]:
        mapping: Dict[str, List[Dict[str, object]]] = {}
        ref_type_map = {"figure": "figure", "table": "table", "section": "heading", "eqn": "formula"}
        for chunk_id in self._ordered_chunk_ids:
            chunk = self.chunk_map.get(chunk_id)
            if not chunk:
                continue
            refs: List[Dict[str, object]] = []
            for match in REFERENCE_RE.finditer(chunk.text or ""):
                label = match.group(1).lower()
                ordinal = int(match.group(2))
                target_key = (ref_type_map.get(label, "text"), ordinal)
                target_chunk = self._ordinal_lookup.get(target_key)
                if not target_chunk:
                    continue
                refs.append({"type": label, "ordinal": ordinal, "target_chunk": target_chunk})
            if refs:
                mapping[chunk_id] = refs
        return mapping

    def semantic_relationships(self, max_neighbors: int = 4) -> List[Dict[str, object]]:
        relationships: List[Dict[str, object]] = []
        embeddings = {chunk_id: self._token_set(self.chunk_map[chunk_id].text) for chunk_id in self._ordered_chunk_ids if chunk_id in self.chunk_map}
        for idx, source_id in enumerate(self._ordered_chunk_ids):
            source_tokens = embeddings.get(source_id)
            if not source_tokens:
                continue
            window = self._ordered_chunk_ids[idx + 1 : idx + 1 + max_neighbors]
            for target_id in window:
                target_tokens = embeddings.get(target_id)
                if not target_tokens:
                    continue
                score = self._jaccard(source_tokens, target_tokens)
                if score < 0.3:
                    continue
                keywords = sorted(source_tokens & target_tokens)[:5]
                relationships.append(
                    {
                        "source": source_id,
                        "target": target_id,
                        "score": round(score, 3),
                        "keywords": keywords,
                        "relation": "shared-keywords",
                    }
                )
        return relationships

    def _ordered_chunks(self) -> List[str]:
        ordered: List[str] = []
        seen = set()
        for section in self.plan.sections:
            if section.heading_chunk_id and section.heading_chunk_id not in seen:
                ordered.append(section.heading_chunk_id)
                seen.add(section.heading_chunk_id)
            for content in section.content:
                if content.chunk_id and content.chunk_id not in seen:
                    ordered.append(content.chunk_id)
                    seen.add(content.chunk_id)
        for chunk_id in self.chunk_map.keys():
            if chunk_id not in seen:
                ordered.append(chunk_id)
        return ordered

    def _build_ordinals(self) -> Dict[Tuple[str, int], str]:
        ordinals: Dict[Tuple[str, int], str] = {}
        counters: Dict[str, int] = {}
        for chunk_id in self._ordered_chunk_ids:
            chunk = self.chunk_map.get(chunk_id)
            if not chunk:
                continue
            region = (chunk.metadata or {}).get("region_type", "text")
            counters[region] = counters.get(region, 0) + 1
            ordinals[(region, counters[region])] = chunk_id
        return ordinals

    def _validation_metrics(
        self,
        node_count: int,
        edge_count: int,
        section_count: int,
        content_count: int,
        covered_chunks: int,
    ) -> Dict[str, object]:
        total_chunks = len(self.chunk_map) or 1
        coverage = round(covered_chunks / total_chunks, 3)
        return {
            "total_nodes": node_count,
            "total_edges": edge_count,
            "section_nodes": section_count,
            "content_nodes": content_count,
            "chunk_coverage": coverage,
        }

    @staticmethod
    def _token_set(text: str) -> set[str]:
        tokens = re.findall(r"[a-zA-Z]{3,}", text or "")
        return {tok.lower() for tok in tokens if tok.lower() not in STOPWORDS}

    @staticmethod
    def _jaccard(a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        intersection = len(a & b)
        union = len(a | b)
        return intersection / union if union else 0.0


def generate_enhanced_graph(
    chunks_path: Path,
    master_plan_path: Path,
    graph_path: Path,
    relationships_path: Path,
    cross_ref_path: Path,
) -> None:
    chunks = common.load_chunks(chunks_path)
    chunk_map = {chunk.chunk_id: chunk for chunk in chunks}
    if not master_plan_path.exists():
        raise FileNotFoundError(master_plan_path)
    from .planner import load_master_plan  # deferred to avoid circular import

    plan = load_master_plan(master_plan_path)
    builder = HierarchicalDocumentGraph(plan, chunk_map)
    graph_payload = builder.build_graph()
    relationships = builder.semantic_relationships()
    cross_refs = builder.cross_reference_map()
    for target in (graph_path, relationships_path, cross_ref_path):
        target.parent.mkdir(parents=True, exist_ok=True)
    graph_path.write_text(json.dumps(graph_payload, indent=2), encoding="utf-8")
    relationships_path.write_text(json.dumps(relationships, indent=2), encoding="utf-8")
    cross_ref_path.write_text(json.dumps(cross_refs, indent=2), encoding="utf-8")


__all__ = ["HierarchicalDocumentGraph", "generate_enhanced_graph"]
