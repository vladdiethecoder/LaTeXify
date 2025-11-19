"""Structure-aware synthesis stage that leverages layout metadata and embeddings."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List

from ..core import common
from ..utils.ensemble import EnsembleVoter
from ..models.math_ocr import MathSyntaxValidator
from .rag import RAGIndex
from .prompt_guard import sanitize_chunk_text
from .specialists import PreambleAgent, SpecialistResult, dispatch_specialist

LOGGER = logging.getLogger(__name__)
RAG_TYPE_MAP = {
    "table": "table",
    "figure": "figure",
    "equation": "equation",
    "list": None,
    "section": None,
    "text": "paragraph",
    "question": "paragraph",
}

DOMAIN_KEYWORDS = {
    "math": ("math", "algebra", "calculus", "geometry", "theorem", "lemma", "proof"),
    "finance": ("finance", "economics", "market", "portfolio", "equity", "bond"),
    "bio": ("biology", "bio", "cell", "genetics", "protein", "enzyme"),
}
PROMPT_GUARD_MARKERS = ("style exemplars", "respond with:", "<<<source")
SYNTAX_VALIDATOR = MathSyntaxValidator()
MARKDOWN_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")


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


def _build_section_context(master_plan_path: Path | None) -> Dict[str, Dict[str, str]]:
    if not master_plan_path or not master_plan_path.exists():
        return {}
    try:
        data = json.loads(master_plan_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    context: Dict[str, Dict[str, str]] = {}
    for section in data.get("sections", []):
        section_id = section.get("section_id")
        section_title = section.get("title") or section_id
        summaries = [item.get("summary", "") for item in section.get("content", []) if item.get("summary")]
        joined_summary = " ".join(summaries)
        if len(joined_summary) > 256:
            joined_summary = joined_summary[:253] + "..."
        for content in section.get("content", []):
            chunk_id = content.get("chunk_id")
            if not chunk_id:
                continue
            context[chunk_id] = {
                "section_id": section_id,
                "section_title": section_title,
                "section_summary": joined_summary,
            }
    return context


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
        heading = markdown_heading or latex_heading(block.label, header_level or 1)
        body = chunk_payload.text.strip()
        combined = "\n".join([heading] + ([body] if body else []))
        return SpecialistResult(latex=combined, notes=notes)
    prefix = ""
    if block.block_type != "section":
        if not conservative_mode and section_context and section_context.get("section_title"):
            prefix += f"% section: {section_context['section_title']}\n"
        if not conservative_mode and parent_section:
            prefix += f"% parent-section: {parent_section}\n"
    heading_prefix = f"{markdown_heading}\n" if markdown_heading and block.block_type != "section" else ""
    snippet_type = RAG_TYPE_MAP.get(block.block_type, block.block_type)
    domain_hint = _infer_domain(parent_section)
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
            refined = refiner.refine(
                block.block_type,
                chunk_payload.text,
                result.latex,
                section_context,
                examples,
            )
            if refined and not _looks_like_refiner_prompt(refined):
                refined_candidate = refined
            elif refined:
                LOGGER.warning(
                    "Discarding refiner prompt echo for chunk %s; using specialist output instead.",
                    chunk.chunk_id,
                )
        except Exception as exc:  # pragma: no cover - heavy dependency
            LOGGER.warning("LLM refinement failed for chunk %s (%s)", chunk.chunk_id, exc)
    threshold = 0.3 if aggressive_mode else 0.4 if conservative_mode else 0.35
    snippet_voter = EnsembleVoter(threshold=threshold)
    snippet_voter.add("specialist", baseline, score=SYNTAX_VALIDATOR.score(baseline))
    if refined_candidate:
        snippet_voter.add(
            "llm-refiner",
            refined_candidate,
            score=SYNTAX_VALIDATOR.score(refined_candidate),
            weight=1.2,
        )
    best_candidate = snippet_voter.best_candidate()
    chosen = (best_candidate.payload if best_candidate else baseline)
    if best_candidate:
        result.notes["snippet_source"] = best_candidate.name
        result.notes["snippet_confidence"] = round(best_candidate.score, 3)
    snippet_body = heading_prefix + chosen if heading_prefix else chosen
    result.latex = prefix + snippet_body
    if "region" not in result.notes:
        result.notes["region"] = region
    if vision_branch:
        result.notes["branch"] = vision_branch
        result.notes["branch_strategy"] = branch_strategy
    return result


def synthesize_blocks(
    plan: Iterable[common.PlanBlock],
    chunk_map: Dict[str, common.Chunk],
    embedding_map: Dict[str, Dict[str, object]],
    graph_map: Dict[str, Dict[str, object]],
    preamble_agent: PreambleAgent,
    rag_index: RAGIndex | None,
    section_context: Dict[str, Dict[str, str]],
    llm_refiner=None,
    quality_profile: Dict[str, object] | None = None,
) -> List[common.Snippet]:
    snippets: List[common.Snippet] = []
    for block in plan:
        chunk = chunk_map.get(block.chunk_id)
        if chunk is None:
            LOGGER.warning("Plan references missing chunk %s", block.chunk_id)
            continue
        embedding = embedding_map.get(block.chunk_id, {})
        graph_context = graph_map.get(block.chunk_id, {})
        section_meta = section_context.get(chunk.chunk_id)
        result = render_block(
            chunk,
            block,
            embedding,
            graph_context,
            preamble_agent,
            rag_index,
            section_meta,
            llm_refiner,
            quality_profile=quality_profile,
        )
        snippets.append(
            common.Snippet(
                chunk_id=chunk.chunk_id,
                latex=result.latex,
                notes=result.notes,
                branch=result.notes.get("branch"),
            )
        )
    return snippets


def run_synthesis(
    chunks_path: Path,
    plan_path: Path,
    graph_path: Path,
    retrieval_path: Path,
    snippets_path: Path,
    preamble_path: Path | None = None,
    document_class: str = "article",
    class_options: str = "11pt",
    rag_index: RAGIndex | None = None,
    master_plan_path: Path | None = None,
    llm_refiner=None,
    quality_profile: Dict[str, object] | None = None,
    domain_profile: Dict[str, object] | None = None,
    refinement_passes: int | None = None,
) -> Path:
    chunks = {chunk.chunk_id: chunk for chunk in common.load_chunks(chunks_path)}
    plan = common.load_plan(plan_path)
    embedding_map: Dict[str, Dict[str, object]] = {}
    if retrieval_path.exists():
        data = json.loads(retrieval_path.read_text(encoding="utf-8"))
        embedding_map = {entry["chunk_id"]: entry for entry in data}
    graph_map: Dict[str, Dict[str, object]] = {}
    if graph_path.exists():
        graph_data = json.loads(graph_path.read_text(encoding="utf-8"))
        parent_lookup = {}
        for edge in graph_data.get("edges", []):
            parent_lookup[edge["target"]] = edge["source"]
        node_lookup = {node["node_id"]: node for node in graph_data.get("nodes", [])}
        for node in graph_data.get("nodes", []):
            chunk_id = (node.get("metadata") or {}).get("chunk_id")
            if chunk_id:
                parent_id = parent_lookup.get(node["node_id"])
                parent_label = node_lookup.get(parent_id, {}).get("label") if parent_id else None
                graph_map[chunk_id] = {"parent_label": parent_label, "node_type": node.get("type")}
    preamble_agent = PreambleAgent()
    section_context = _build_section_context(master_plan_path)
    snippets = synthesize_blocks(
        plan,
        chunks,
        embedding_map,
        graph_map,
        preamble_agent,
        rag_index,
        section_context,
        llm_refiner,
        quality_profile=quality_profile,
    )
    common.save_snippets(snippets, snippets_path)
    if preamble_path:
        payload = {
            "document_class": document_class,
            "class_options": class_options,
            "packages": preamble_agent.packages(),
        }
        if quality_profile:
            payload["quality_mode"] = quality_profile.get("processing_mode")
        if domain_profile:
            payload["domain_profile"] = domain_profile
        preamble_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        LOGGER.info("PreambleAgent registered %s packages", len(payload["packages"]))
    LOGGER.info("Synthesis complete with %s structurally-aware snippets", len(snippets))
    return snippets_path


__all__ = ["run_synthesis", "synthesize_blocks"]
