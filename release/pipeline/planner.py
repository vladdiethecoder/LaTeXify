"""PlannerAgent that emits a schema-constrained master plan."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional

from pydantic import BaseModel, Field

from ..core import common
from .enhanced_structure_graph import generate_enhanced_graph

LOGGER = logging.getLogger(__name__)
ContentType = Literal["paragraph", "equation", "table", "list", "figure", "metadata"]


class PlanContent(BaseModel):
    item_id: str
    chunk_id: str
    type: ContentType
    summary: str | None = None
    region_type: str | None = None
    header_level: int | None = None


class PlanSection(BaseModel):
    section_id: str
    title: str
    header_level: int = 1
    heading_chunk_id: str | None = None
    content: List[PlanContent] = Field(default_factory=list)


class MasterPlan(BaseModel):
    document_title: str
    document_class: str = "article"
    class_options: str = "12pt,twocolumn"
    sections: List[PlanSection]


def _safe_label(text: str, fallback: str) -> str:
    stripped = (text or "").strip()
    if not stripped:
        return fallback
    first_line = stripped.splitlines()[0].strip()
    return first_line or fallback


def _summarize(text: str, limit: int = 32) -> str:
    tokens = [tok.strip() for tok in text.replace("\n", " ").split() if tok.strip()]
    if not tokens:
        return ""
    if len(tokens) <= limit:
        return " ".join(tokens)
    return " ".join(tokens[:limit]) + "..."


def _content_type(region_type: str) -> ContentType:
    mapping: Dict[str, ContentType] = {
        "table": "table",
        "list": "list",
        "figure": "figure",
        "formula": "equation",
        "heading": "metadata",
    }
    return mapping.get(region_type, "paragraph")


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
    document_class: str = "article",
    class_options: str = "12pt,twocolumn",
) -> MasterPlan:
    sections: List[PlanSection] = []
    section_counter = 1
    current_section: Optional[PlanSection] = None

    def ensure_current(title: str, header_level: int) -> PlanSection:
        nonlocal current_section, section_counter
        current_section = _ensure_section(sections, section_counter, title, header_level)
        section_counter += 1
        return current_section

    for chunk in chunks:
        metadata = chunk.metadata or {}
        header_level = metadata.get("header_level", 0)
        region = metadata.get("region_type", "text")
        if header_level > 0 or region == "heading":
            title = _safe_label(chunk.text, f"Section {section_counter}")
            section = ensure_current(title, header_level or 1)
            section.heading_chunk_id = chunk.chunk_id
            continue
        if current_section is None:
            ensure_current("Introduction", 1)
        assert current_section is not None  # mypy guard
        current_section.content.append(
            PlanContent(
                item_id=f"{current_section.section_id}-content-{len(current_section.content) + 1:03d}",
                chunk_id=chunk.chunk_id,
                type=_content_type(region),
                summary=_summarize(chunk.text),
                region_type=region,
                header_level=metadata.get("header_level"),
            )
        )

    if not sections:
        default_section = PlanSection(section_id="sec-001", title=document_title or "Document", header_level=1)
        sections.append(default_section)

    return MasterPlan(
        document_title=document_title or "Generated Document",
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
    document_class: str = "article",
    class_options: str = "12pt,twocolumn",
) -> Path:
    chunks = common.load_chunks(chunks_path)
    plan = build_master_plan(chunks, document_title or "Generated Document", document_class, class_options)
    save_master_plan(plan, master_plan_path)
    LOGGER.info("PlannerAgent generated %s sections", len(plan.sections))
    try:
        output_dir = master_plan_path.parent
        generate_enhanced_graph(
            chunks_path,
            master_plan_path,
            output_dir / "enhanced_structure_graph.json",
            output_dir / "semantic_relationships.json",
            output_dir / "cross_reference_map.json",
        )
    except Exception as exc:  # pragma: no cover - graph is auxiliary
        LOGGER.warning("Enhanced structure graph generation skipped: %s", exc)
    return master_plan_path


__all__ = ["MasterPlan", "PlanSection", "PlanContent", "run_planner", "load_master_plan", "build_master_plan"]
