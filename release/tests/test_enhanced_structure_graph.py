import json
from pathlib import Path

from release.core import common
from release.pipeline.planner import MasterPlan, PlanSection, PlanContent, save_master_plan
from release.pipeline.enhanced_structure_graph import generate_enhanced_graph


def _chunk(chunk_id: str, page: int, text: str, region: str, **metadata):
    meta = {"region_type": region}
    meta.update(metadata)
    return common.Chunk(chunk_id=chunk_id, page=page, text=text, metadata=meta)


def test_generate_enhanced_graph(tmp_path):
    chunks = [
        _chunk("c1", 1, "Section 1 introduction", "heading", layout_confidence=0.9),
        _chunk("c2", 1, "Figure 1 shows the setup.", "figure", layout_confidence=0.8),
        _chunk("c3", 2, "Section 2 elaborates on Figure 1.", "text", layout_confidence=0.75),
    ]
    chunks_path = tmp_path / "chunks.json"
    common.save_chunks(chunks, chunks_path)
    plan = MasterPlan(
        document_title="Demo",
        document_class="article",
        class_options="12pt",
        sections=[
            PlanSection(
                section_id="sec-001",
                title="Intro",
                header_level=1,
                heading_chunk_id="c1",
                content=[
                    PlanContent(item_id="item-1", chunk_id="c2", type="figure"),
                    PlanContent(item_id="item-2", chunk_id="c3", type="paragraph"),
                ],
            )
        ],
    )
    plan_path = tmp_path / "plan.json"
    save_master_plan(plan, plan_path)
    enhanced_graph = tmp_path / "enhanced_structure_graph.json"
    relationships = tmp_path / "semantic_relationships.json"
    cross_map = tmp_path / "cross_reference_map.json"
    generate_enhanced_graph(chunks_path, plan_path, enhanced_graph, relationships, cross_map)
    graph_payload = json.loads(enhanced_graph.read_text(encoding="utf-8"))
    assert graph_payload["metrics"]["section_nodes"] == 1
    rel_payload = json.loads(relationships.read_text(encoding="utf-8"))
    assert isinstance(rel_payload, list)
    cross_payload = json.loads(cross_map.read_text(encoding="utf-8"))
    assert isinstance(cross_payload, dict)
