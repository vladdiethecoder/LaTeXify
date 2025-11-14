import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from release.pipeline import assembly, ingestion, layout, planner, rag, retrieval, structure_graph, synthesis

SAMPLE_PDF = Path(__file__).resolve().parents[1] / "samples" / "sample.pdf"


def test_ingestion_and_layout(tmp_path):
    result = ingestion.run_ingestion(SAMPLE_PDF, tmp_path / "artifacts", chunk_chars=400, ocr_mode="none", capture_page_images=False)
    assert result.chunks_path.exists()
    master_plan_path = tmp_path / "master_plan.json"
    planner.run_planner(result.chunks_path, master_plan_path, document_title="Sample")
    plan_path = tmp_path / "plan.json"
    layout.run_layout(result.chunks_path, plan_path, master_plan_path)
    assert plan_path.exists()


def test_synthesis_and_assembly(tmp_path):
    result = ingestion.run_ingestion(SAMPLE_PDF, tmp_path / "artifacts", chunk_chars=400, ocr_mode="none", capture_page_images=False)
    master_plan_path = tmp_path / "master_plan.json"
    planner.run_planner(result.chunks_path, master_plan_path, document_title="Sample")
    master_plan = planner.load_master_plan(master_plan_path)
    plan_path = tmp_path / "plan.json"
    layout.run_layout(result.chunks_path, plan_path, master_plan_path)
    graph_path = tmp_path / "graph.json"
    structure_graph.build_graph(plan_path, result.chunks_path, graph_path)
    retrieval_path = tmp_path / "retrieval.json"
    retrieval.build_index(result.chunks_path, plan_path, retrieval_path)
    snippets_path = tmp_path / "snippets.json"
    preamble_path = tmp_path / "preamble.json"
    rag_index = rag.RAGIndex([])
    synthesis.run_synthesis(
        result.chunks_path,
        plan_path,
        graph_path,
        retrieval_path,
        snippets_path,
        preamble_path,
        master_plan.document_class,
        master_plan.class_options,
        rag_index,
        llm_refiner=None,
    )
    tex_path = assembly.run_assembly(
        plan_path,
        snippets_path,
        preamble_path,
        tmp_path / "output",
        title="Test Doc",
        author="Unit Test",
        skip_compile=True,
    )
    assert tex_path.exists()
    content = tex_path.read_text(encoding="utf-8")
    assert "\\documentclass" in content


def test_master_plan_schema(tmp_path):
    result = ingestion.run_ingestion(SAMPLE_PDF, tmp_path / "artifacts", chunk_chars=400, ocr_mode="none", capture_page_images=False)
    master_plan_path = tmp_path / "master_plan.json"
    planner.run_planner(result.chunks_path, master_plan_path, document_title="Sample Title")
    data = json.loads(master_plan_path.read_text(encoding="utf-8"))
    assert data["document_title"] == "Sample Title"
    assert "sections" in data and len(data["sections"]) > 0
    first_section = data["sections"][0]
    assert "section_id" in first_section
    assert "content" in first_section
