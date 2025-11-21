"""
Semantic Graph-RAG for Cross-References.
Builds a NetworkX graph of document entities (Equations, Figures, Sections) to resolve references.
"""
import re
import networkx as nx
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class DocumentEntity:
    id: str
    type: str # equation, figure, table, section
    content: str
    page: int

class SemanticGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.label_map: Dict[str, str] = {} # label -> node_id

    def add_node(self, entity: DocumentEntity):
        self.graph.add_node(entity.id, data=entity)
        # Parse labels from content
        labels = re.findall(r'\\label{([^}]+)}', entity.content)
        for label in labels:
            self.label_map[label] = entity.id

    def build_edges(self):
        """Link nodes based on \\ref{} calls."""
        for node_id, data in self.graph.nodes(data=True):
            entity = data['data']
            refs = re.findall(r'\\ref{([^}]+)}', entity.content)
            for ref in refs:
                if ref in self.label_map:
                    target_id = self.label_map[ref]
                    self.graph.add_edge(node_id, target_id, type="refers_to")

    def get_context(self, chunk_text: str) -> str:
        """
        Retrieve definitions for references found in the chunk text.
        """
        refs = re.findall(r'\\ref{([^}]+)}', chunk_text)
        context_snippets = []
        
        for ref in refs:
            if ref in self.label_map:
                target_id = self.label_map[ref]
                entity = self.graph.nodes[target_id]['data']
                context_snippets.append(f"% Reference Context ({entity.type}):\n{entity.content}")
                
        return "\n\n".join(context_snippets)

# Integration helper
def process_references(chunks: List[Dict]) -> SemanticGraph:
    sg = SemanticGraph()
    # First pass: nodes
    for i, chunk in enumerate(chunks):
        # Simple heuristic: Treat chunk as entity if it has a label
        if "\\label" in chunk['text']:
            entity = DocumentEntity(
                id=f"chunk_{i}",
                type=chunk.get("metadata", {}).get("region_type", "text"),
                content=chunk['text'],
                page=chunk.get("page", 0)
            )
            sg.add_node(entity)
    
    # Second pass: edges
    sg.build_edges()
    return sg
