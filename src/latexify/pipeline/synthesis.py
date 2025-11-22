"""Structure-aware synthesis stage that leverages layout metadata and embeddings."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Any

from ..core import common
from latexify.core.state import DocumentState
from ..utils.ensemble import EnsembleVoter
from ..models.math_ocr import MathSyntaxValidator
from .rag import RAGIndex
from .prompt_guard import sanitize_chunk_text
from .specialists import PreambleAgent, SpecialistResult, dispatch_specialist
from . import assembly # Import assembly for final stitching

LOGGER = logging.getLogger(__name__)
RAG_TYPE_MAP = {
    "table": "table",
    "figure": "figure",
    "equation": "equation",
    "list": None,
    "section": None,
    "text": "paragraph",
    "question": "paragraph",
    "question_block": "paragraph",
    "answer_block": "paragraph",
    "display_equation": "equation",
}

DOMAIN_KEYWORDS = {
    "math": ("math", "algebra", "calculus", "geometry", "theorem", "lemma", "proof"),
    "finance": ("finance", "economics", "market", "portfolio", "equity", "bond"),
    "bio": ("biology", "bio", "cell", "genetics", "protein", "enzyme"),
}
PROMPT_GUARD_MARKERS = ("style exemplars", "respond with:", "<<<source")
SYNTAX_VALIDATOR = MathSyntaxValidator()
MARKDOWN_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")


def synthesis_node(state: DocumentState) -> DocumentState:
    """
    Synthesis Node: Converts Semantic Plan + Chunks -> LaTeX Source.
    Combines Specialist Synthesis and Assembly.
    """
    LOGGER.info("Starting Synthesis Node...")
    
    # 1. Prepare Inputs
    if not state.semantic_plan:
        LOGGER.warning("No semantic plan found.")
        state.generated_latex = "% No plan available"
        return state
        
    # Adapt plan dict back to PlanBlock objects
    plan_blocks = _adapt_plan_from_state(state.semantic_plan)
    
    chunk_map = {c.chunk_id: c for c in state.chunks}
    
    preamble_agent = PreambleAgent()
    section_context = _build_section_context_from_dict(state.semantic_plan)
    
    # 2. Generate Snippets
    snippets = []
    for block in plan_blocks:
        chunk = chunk_map.get(block.chunk_id)
        if block.block_type == "section" and not chunk:
             chunk = common.Chunk(chunk_id=block.chunk_id or "gen", page=0, text=block.label)
             
        if not chunk:
            continue
            
        # Get examples from state
        examples_data = state.reference_snippets.get(chunk.chunk_id, [])
        from .rag import RAGEntry
        examples = [RAGEntry.from_json(e) for e in examples_data]
        
        result = render_block(
            chunk=chunk,
            block=block,
            embedding={},
            graph_context={},
            preamble_agent=preamble_agent,
            rag_index=None,
            section_context=section_context.get(chunk.chunk_id),
            llm_refiner=None, # Phase 4 handles refinement separately
            quality_profile={"processing_mode": "conservative"},
            override_examples=examples
        )
        
        snippets.append(common.Snippet(
            chunk_id=chunk.chunk_id,
            latex=result.latex,
            notes=result.notes,
            branch=result.notes.get("branch")
        ))
    
    # 3. Assembly (Generate main.tex content)
    # assembly.build_preamble now expects a dict with 'packages'
    # We must ensure we are merging the preamble agent packages with assembly's defaults if needed,
    # OR we trust assembly.build_preamble to handle it if we pass partial config.
    # Looking at assembly.py, `build_preamble` takes a config dict.
    # It does NOT auto-merge BASE_PACKAGES unless we use `load_preamble_config`.
    # But here we are calling `build_preamble` directly with a constructed dict.
    
    # Let's manually merge BASE_PACKAGES from assembly to be safe, or rely on PreambleAgent to have them?
    # PreambleAgent has some (graphicx, geometry, float). But NOT siunitx by default unless requested.
    # The prompt requests structure-aware refinement which implies we might need more.
    # assembly.BASE_PACKAGES has siunitx.
    
    from .assembly import BASE_PACKAGES, _ensure_base_packages
    
    # Merge PreambleAgent packages with BASE_PACKAGES
    agent_packages = preamble_agent.packages()
    merged_packages = _ensure_base_packages(agent_packages)
    
    preamble_config = {
        "document_class": "article",
        "packages": merged_packages
    }
    preamble = assembly.build_preamble(preamble_config)
    
    snippet_map = {s.chunk_id: s.latex for s in snippets}
    used_assets = set()
    assets_dir = state.file_path.parent / "assets" if state.file_path else Path("assets") 
    
    body_parts = []
    for block in plan_blocks:
        latex = assembly.block_to_latex(
            block,
            snippet_map,
            assets_dir,
            used_assets,
            chunk_lookup=chunk_map,
            document_class="article"
        )
        body_parts.append(latex)
        
    full_latex = "\n".join([
        preamble,
        f"\\title{{{state.document_name}}}",
        "\\maketitle",
        *body_parts,
        "\\end{document}"
    ])
    
    state.generated_latex = full_latex
    LOGGER.info("Synthesis complete. LaTeX generated.")
    return state


def _adapt_plan_from_state(plan_dict: Dict[str, Any]) -> List[common.PlanBlock]:
    blocks = []
    if "sections" in plan_dict:
        for section in plan_dict["sections"]:
             # Section Heading
            blocks.append(common.PlanBlock(
                block_id=section.get("section_id"),
                chunk_id=section.get("heading_chunk_id") or "",
                label=section.get("title", ""),
                block_type="section",
                metadata={"hierarchy_level": section.get("header_level", 1)}
            ))
            for item in section.get("content", []):
                blocks.append(common.PlanBlock(
                    block_id=item.get("item_id"),
                    chunk_id=item.get("chunk_id"),
                    label="",
                    block_type=item.get("type"),
                    metadata={
                        "summary": item.get("summary"),
                        "bbox": item.get("layout_bbox")
                    }
                ))
    return blocks

def _build_section_context_from_dict(plan: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    context = {}
    for section in plan.get("sections", []):
        s_id = section.get("section_id")
        s_title = section.get("title")
        for item in section.get("content", []):
             c_id = item.get("chunk_id")
             if c_id:
                 context[c_id] = {"section_id": s_id, "section_title": s_title}
    return context


def _infer_domain(label: str | None) -> str | None:
    if not label:
        return None
    normalized = label.lower()
    for domain, keywords in DOMAIN_KEYWORDS.items():
        if any(keyword in normalized for keyword in keywords):
            return domain
    return None


def latex_heading(label: str, level: int) -> str:
    if level <= 1:
        return f"\\section{{{label}}}"
    if level == 2:
        return f"\\subsection{{{label}}}"
    if level == 3:
        return f"\\subsubsection{{{label}}}"
    return f"\\paragraph{{{label}}}"


def _extract_markdown_heading(text: str) -> tuple[str | None, int | None, str]:
    lines = text.splitlines()
    heading_cmd: str | None = None
    heading_level: int | None = None
    remainder_start = 0
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        match = MARKDOWN_HEADING_RE.match(stripped)
        if match:
            heading_level = len(match.group(1))
            label = match.group(2).strip()
            heading_cmd = latex_heading(label, heading_level)
            remainder_start = idx + 1
        break
    remainder = "\n".join(lines[remainder_start:]).lstrip("\n")
    if heading_cmd:
        return heading_cmd, heading_level, remainder
    return None, None, text


def _looks_like_refiner_prompt(candidate: str | None) -> bool:
    if not candidate:
        return False
    lowered = candidate.lower()
    return sum(marker in lowered for marker in PROMPT_GUARD_MARKERS) >= 2


def _branch_mode(profile: Dict[str, object] | None) -> str:
    if not profile:
        return "ocr-only"
    return str(profile.get("branch_strategy") or "ocr-only")


def _fuse_branch_text(
    text: str,
    branch_info: Dict[str, object] | None,
    strategy: str,
) -> str:
    if not branch_info or strategy == "ocr-only":
        return text
    hints: List[str] = []
    region = branch_info.get("region_type")
    if region:
        hints.append(f"region={region}")
    bbox = branch_info.get("bbox")
    if bbox:
        hints.append("bbox=" + ",".join(str(value) for value in bbox))
    if branch_info.get("page_image"):
        hints.append(f"page_image={branch_info['page_image']}")
    branch_id = branch_info.get("branch_id")
    if branch_id:
        hints.append(f"id={branch_id}")
    prefix = "% vision-branch " + " | ".join(hints)
    return prefix + "\n" + text


def render_block(
    chunk: common.Chunk,
    block: common.PlanBlock,
    embedding: Dict[str, object],
    graph_context: Dict[str, object],
    preamble_agent: PreambleAgent,
    rag_index: RAGIndex | None,
    section_context: Dict[str, str] | None,
    llm_refiner=None,
    quality_profile: Dict[str, object] | None = None,
    # ADDED override
    override_examples: List[Any] | None = None
) -> SpecialistResult:
    metadata = chunk.metadata or {}
    branch_provenance = metadata.get("branch_provenance") or {}
    vision_branch = branch_provenance.get("vision") if isinstance(branch_provenance, dict) else None
    branch_strategy = _branch_mode(quality_profile)
    region = metadata.get("region_type", "text")
    if vision_branch and branch_strategy != "ocr-only":
        region = vision_branch.get("region_type", region)
    header_level = metadata.get("header_level", 0)
    parent_section = graph_context.get("parent_label")
    notes = {"region": region}
    if section_context:
        notes.update(section_context)
    guarded_text = sanitize_chunk_text(chunk.text)
    guarded_text = _fuse_branch_text(guarded_text, vision_branch, branch_strategy)
    conservative_mode = (quality_profile or {}).get("processing_mode") == "conservative"
    aggressive_mode = (quality_profile or {}).get("processing_mode") == "aggressive"
    if conservative_mode:
        markdown_heading, normalized_heading, normalized_text = None, None, guarded_text
    else:
        markdown_heading, _, normalized_text = _extract_markdown_heading(guarded_text)
    chunk_payload = chunk
    if normalized_text != chunk.text:
        chunk_payload = common.Chunk(
            chunk_id=chunk.chunk_id,
            page=chunk.page,
            text=normalized_text,
            images=chunk.images,
            metadata=chunk.metadata,
        )
    if block.block_type == "section":
        title = block.label
        if not title and chunk_payload:
             title = chunk_payload.text.strip()
        level = (block.metadata or {}).get("hierarchy_level", 1)
        heading = latex_heading(title, level)
        return SpecialistResult(latex=heading, notes=notes)

    prefix = ""
    if block.block_type != "section":
        if not conservative_mode and section_context and section_context.get("section_title"):
            prefix += f"% section: {section_context['section_title']}\n"
        if not conservative_mode and parent_section:
            prefix += f"% parent-section: {parent_section}\n"
    heading_prefix = f"{markdown_heading}\n" if markdown_heading and block.block_type != "section" else ""
    snippet_type = RAG_TYPE_MAP.get(block.block_type, block.block_type)
    domain_hint = _infer_domain(parent_section)
    
    if override_examples is not None:
        examples = override_examples
    else:
        examples = (
            rag_index.search(chunk_payload.text, snippet_type, k=2, domain=domain_hint)
            if rag_index and snippet_type
            else []
        )
        
    result = dispatch_specialist(block.block_type, chunk_payload, preamble_agent, examples, context=section_context)
    baseline = result.latex
    refined_candidate = None
    refiner = llm_refiner if not conservative_mode else None
    if refiner:
        try:
            refined = refiner.refine(result.latex) 
            if refined and not _looks_like_refiner_prompt(refined):
                refined_candidate = refined
            elif refined:
                LOGGER.warning(
                    "Discarding refiner prompt echo for chunk %s; using specialist output instead.",
                    chunk.chunk_id,
                )
        except Exception as exc:
            LOGGER.warning("LLM refinement failed for chunk %s (%s)", chunk.chunk_id, exc)
    
    if refined_candidate:
        chosen = refined_candidate
        result.notes["snippet_source"] = "llm-refiner"
    else:
        chosen = baseline
        result.notes["snippet_source"] = "specialist"

    snippet_body = heading_prefix + chosen if heading_prefix else chosen
    result.latex = prefix + snippet_body
    if "region" not in result.notes:
        result.notes["region"] = region
    if vision_branch:
        result.notes["branch"] = vision_branch
        result.notes["branch_strategy"] = branch_strategy
    return result

def run_synthesis(*args, **kwargs):
    # Legacy stub
    pass

def synthesize_blocks(*args, **kwargs):
    # Legacy stub
    pass

__all__ = ["run_synthesis", "synthesize_blocks", "synthesis_node"]