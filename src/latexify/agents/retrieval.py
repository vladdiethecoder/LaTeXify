"""Retrieval Agent Node."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Any

from latexify.core.state import DocumentState
from latexify.core import common
from latexify.pipeline.rag import load_or_build_index, RAGIndex, RAGEntry

LOGGER = logging.getLogger(__name__)

def retrieve_node(state: DocumentState) -> DocumentState:
    """
    Retrieval Node: Populates reference_snippets using RAG.
    """
    LOGGER.info("Starting Retrieval Node...")
    
    # 1. Setup RAG Index
    # TODO: Make paths configurable via state.config
    # Assuming repo layout is consistent
    repo_root = Path(__file__).resolve().parents[3] # src/latexify/agents/ -> src/latexify/ -> src/ -> root
    if not (repo_root / "release").exists():
        # Fallback if running from within release dir or unexpected layout
        # Try to find 'release' by walking up
        repo_root = Path.cwd()
        if not (repo_root / "release").exists():
            if (repo_root / "reference_tex").exists():
                 # We are likely IN release/
                 rag_source = repo_root / "reference_tex"
                 rag_cache = repo_root / "cache" / "rag_index.json"
            else:
                 LOGGER.warning("Could not locate RAG source. Skipping retrieval.")
                 rag_source = None
        else:
            rag_source = repo_root / "release" / "reference_tex"
            rag_cache = repo_root / "release" / "cache" / "rag_index.json"
    else:
        rag_source = repo_root / "release" / "reference_tex"
        rag_cache = repo_root / "release" / "cache" / "rag_index.json"

    if not rag_source or not rag_source.exists():
        LOGGER.warning(f"RAG source not found. Retrieval will be empty.")
        rag_index = RAGIndex([])
    else:
        rag_index = load_or_build_index(rag_source, rag_cache)
    
    # 2. Prepare Chunks
    chunk_map = {c.chunk_id: c for c in state.chunks}
    
    # 3. Iterate Plan
    plan = state.semantic_plan or {}
    sections = plan.get("sections", [])
    
    count = 0
    for section in sections:
        for content in section.get("content", []):
            chunk_id = content.get("chunk_id")
            content_type = content.get("type", "paragraph")
            
            if chunk_id and chunk_id in chunk_map:
                chunk = chunk_map[chunk_id]
                # Search
                snippet_type = _map_rag_type(content_type)
                
                results = rag_index.search(chunk.text, snippet_type, k=2)
                if results:
                    state.reference_snippets[chunk_id] = [r.to_json() for r in results]
                    count += 1
                    
    LOGGER.info(f"Retrieval complete. Context found for {count} blocks.")
    return state

def _map_rag_type(plan_type: str) -> str | None:
    if plan_type in ["table"]:
        return "table"
    if plan_type in ["figure"]:
        return "figure"
    if plan_type in ["equation", "display_equation"]:
        return "equation"
    # Default or None for generic text
    return None