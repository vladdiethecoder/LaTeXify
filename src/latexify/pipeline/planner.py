"PlannerAgent that emits a schema-constrained master plan based on semantic tags."
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional

from pydantic import BaseModel, Field

from ..core import common
from latexify.core.state import DocumentState
# from .enhanced_structure_graph import generate_enhanced_graph

LOGGER = logging.getLogger(__name__)

# Extended types to match Blueprint
ContentType = Literal[
    "paragraph", 
    "display_equation", 
    "table", 
    "list", 
    "figure", 
    "metadata",
    "question_block",
    "answer_block",
    "theorem_block",
    "proof_block"
]


class PlanContent(BaseModel):
    item_id: str
    chunk_id: str
    type: ContentType
    summary: str | None = None
    # Carry forward layout data for synthesis agents
    layout_bbox: List[float] | None = None
    contains_images: bool = False
    contains_equations: bool = False


class PlanSection(BaseModel):
    section_id: str
    title: str
    header_level: int = 1
    heading_chunk_id: str | None = None
    content: List[PlanContent] = Field(default_factory=list)


class MasterPlan(BaseModel):
    document_title: str
    document_class: str = "article" # Default, usually overridden by CLI
    class_options: str = "12pt,twoside" # Better default for textbooks
    sections: List[PlanSection]


def plan_node(state: DocumentState) -> DocumentState:
    """
    Planning Node: Generates a MasterPlan from semantic chunks.
    """
    LOGGER.info("Starting Planning Node...")
    if not state.chunks:
        LOGGER.warning("No chunks found in state. Planning might be empty.")
    
    # Reuse existing logic
    plan = build_master_plan(
        chunks=state.chunks,
        document_title=state.document_name,
        document_class="article" # could be from state.config
    )
    
    state.semantic_plan = plan.model_dump()
    LOGGER.info(f"Planning complete. Generated {len(plan.sections)} sections.")
    return state


def _safe_label(text: str, fallback: str) -> str:
    stripped = (text or "").strip()
    if not stripped:
        return fallback
    # Extract first line or reasonable substring
    lines = stripped.splitlines()
    if not lines:
        return fallback
    return lines[0][:100].strip() or fallback


def _summarize(text: str, limit: int = 32) -> str:
    tokens = [tok.strip() for tok in text.replace("\n", " ").split() if tok.strip()]
    if not tokens:
        return ""
    if len(tokens) <= limit:
        return " ".join(tokens)
    return " ".join(tokens[:limit]) + "..."


def _map_tag_to_type(tag: str, chunk: common.Chunk) -> ContentType:
    """
    Maps ingestion semantic tags to synthesis templates.
    """
    # 1. Strong types from ingestion
    if tag == "question":
        return "question_block"
    if tag == "answer":
        return "answer_block"
    if tag == "equation":
        return "display_equation"
    if tag == "figure":
        return "figure"
    if tag == "table":
        return "table"
    
    # 2. Text heuristics (if ingestion missed it)
    text_lower = chunk.text.lower().strip()
    if text_lower.startswith("proof"):
        return "proof_block"
    if text_lower.startswith("theorem") or text_lower.startswith("lemma"):
        return "theorem_block"
        
    # 3. Content scanning
    # If text contains "where x is...", it's a paragraph. 
    return "paragraph"


def _ensure_section(
    sections: List[PlanSection],
    section_counter: int,
    title: str,
    header_level: int,
) -> PlanSection:
    section = PlanSection(
        section_id=f"sec-{section_counter:03d}",
        title=title or f"Section {section_counter}",
        header_level=max(1, header_level or 1),
    )
    sections.append(section)
    return section


def build_master_plan(
    chunks: Iterable[common.Chunk],
    document_title: str,
    document_class: str = "book", # Default to book for textbook quality
    class_options: str = "12pt",
) -> MasterPlan:
    sections: List[PlanSection] = []
    section_counter = 1
    current_section: Optional[PlanSection] = None

    def ensure_current(title: str, header_level: int) -> PlanSection:
        nonlocal current_section, section_counter
        current_section = _ensure_section(sections, section_counter, title, header_level)
        section_counter += 1
        return current_section

    # Initial pass: Create a default section if none exists
    # ensure_current("Introduction", 1)

    for chunk in chunks:
        metadata = chunk.metadata or {}
        tag = metadata.get("tag", "text")
        
        # 1. Handle Section Headings
        if tag == "heading":
            title = _safe_label(chunk.text, f"Section {section_counter}")
            # Ingest might not give level, assume 1 for now or infer from font size if available
            header_level = 1 
            section = ensure_current(title, header_level)
            section.heading_chunk_id = chunk.chunk_id
            continue
            
        # 2. Ensure we have a section context
        if current_section is None:
            ensure_current("Preamble / Introduction", 1)
        assert current_section is not None

        # 3. Map Content
        plan_type = _map_tag_to_type(tag, chunk)
        
        # 4. Create Node
        node = PlanContent(
            item_id=f"{current_section.section_id}-item-{len(current_section.content) + 1:03d}",
            chunk_id=chunk.chunk_id,
            type=plan_type,
            summary=_summarize(chunk.text),
            layout_bbox=metadata.get("bbox"),
            contains_images=metadata.get("contains_images", False),
            contains_equations=metadata.get("contains_equations", False)
        )
        current_section.content.append(node)

    return MasterPlan(
        document_title=document_title or "Generated Textbook",
        document_class=document_class,
        class_options=class_options,
        sections=sections,
    )


def save_master_plan(plan: MasterPlan, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_master_plan(path: Path) -> MasterPlan:
    data = json.loads(path.read_text(encoding="utf-8"))
    return MasterPlan.model_validate(data)


def run_planner(
    chunks_path: Path,
    master_plan_path: Path,
    document_title: str | None = None,
    document_class: str = "book",
    class_options: str = "12pt",
) -> Path:
    chunks = common.load_chunks(chunks_path)
    plan = build_master_plan(chunks, document_title or "Generated Document", document_class, class_options)
    save_master_plan(plan, master_plan_path)
    LOGGER.info("PlannerAgent generated %s sections", len(plan.sections))
    
    # Optional: Generate Graph (commented out to reduce dependencies for this phase)
    # try:
    #     output_dir = master_plan_path.parent
    #     generate_enhanced_graph(...)
    # except Exception as exc:
    #     LOGGER.warning("Enhanced structure graph generation skipped: %s", exc)
        
    return master_plan_path


__all__ = ["MasterPlan", "PlanSection", "PlanContent", "run_planner", "load_master_plan", "build_master_plan", "plan_node"]