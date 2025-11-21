import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from latexify.core import common
from latexify.pipeline import active_learning


def test_active_learning_queue_builds_records(tmp_path):
    chunks = [
        common.Chunk(chunk_id="chunk-1", page=1, text="Question text", metadata={"region_type": "equation"}),
        common.Chunk(chunk_id="chunk-2", page=2, text="Answer text", metadata={"region_type": "text"}),
    ]
    common.save_chunks(chunks, tmp_path / "chunks.json")
    plan_blocks = [
        common.PlanBlock(block_id="b1", chunk_id="chunk-1", label="Q1", block_type="equation"),
        common.PlanBlock(block_id="b2", chunk_id="chunk-2", label="Q1 Answer", block_type="text"),
    ]
    common.save_plan(plan_blocks, tmp_path / "plan.json")
    snippets = [common.Snippet(chunk_id="chunk-1", latex="\\begin{align}x=1\\end{align}")]
    common.save_snippets(snippets, tmp_path / "snippets.json")

    quality_report = {"aggregate": 0.55, "weak_sections": ["chunk-1"]}
    hallucination_report = {"flagged": [{"chunk_id": "chunk-1", "heading": "Bad"}]}
    gaps_report = {"missing_chunk_ids": ["chunk-2"], "expected_snippets": 2, "actual_snippets": 1}
    visual_report = {
        "available": True,
        "records": [{"page": 1, "difference": 0.4, "status": "flagged"}],
        "flagged_pages": 1,
        "pages_evaluated": 1,
    }
    reward_report = {"reward": -0.2, "components": {"syntax": -1.0}}
    validation_report = {"success": False, "errors": ["Undefined control sequence"]}
    lint_report = {"issues": ["Spacing"], "notes": "1 warning"}

    result = active_learning.build_active_learning_queue(
        run_id="unit-test",
        chunks_path=tmp_path / "chunks.json",
        plan_path=tmp_path / "plan.json",
        snippets_path=tmp_path / "snippets.json",
        output_dir=tmp_path,
        quality_report=quality_report,
        hallucination_report=hallucination_report,
        gaps_report=gaps_report,
        visual_report=visual_report,
        reward_report=reward_report,
        validation_report=validation_report,
        lint_report=lint_report,
        limit=10,
    )

    assert result.summary["total_candidates"] == 3
    queue_records = [json.loads(line) for line in result.queue_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert any(rec["chunk_id"] == "chunk-1" and "visual_regression" in rec["reasons"] for rec in queue_records)
    assert any(rec["chunk_id"] == "chunk-2" and "missing_snippet" in rec["reasons"] for rec in queue_records)
    assert any(rec["chunk_id"] == active_learning.DOCUMENT_CHUNK_ID and "validation_error" in rec["reasons"] for rec in queue_records)
    assert result.summary_path.exists()
    assert "low_quality" in result.summary["reason_counts"]
