# tests/test_retrieval_bundle.py
from __future__ import annotations
import json
from pathlib import Path

from latexify.pipeline.retrieval_bundle import build_context_bundle

def test_bundle_shapes_without_indexes(tmp_path: Path):
    # Minimal plan and task
    plan = {
        "tasks": [
            {"id": "T01", "order": 1, "anchor": "frontmatter.title", "title": "Title & Metadata"},
            {"id": "T03", "order": 3, "anchor": "introduction", "title": "Introduction"},
        ]
    }
    (tmp_path / "plan.json").write_text(json.dumps(plan), encoding="utf-8")

    # Provide indices that don't exist (graceful empty retrieval)
    indices = {
        "assignment": tmp_path / "missing_assignment",
        "assessment": tmp_path / "missing_assessment",
        "rubric": tmp_path / "missing_rubric",
        "user": tmp_path / "missing_user",
    }
    task = plan["tasks"][1]
    bundle = build_context_bundle(task, indices, evidence_dir=tmp_path / "evidence")

    assert bundle.task_id == "T03"
    assert isinstance(bundle.rubric, list)
    assert isinstance(bundle.assignment_rules, list)
    assert isinstance(bundle.assessment, list)
    assert isinstance(bundle.user_answer.chunks, list)
    assert "ocr_uncertain" in bundle.user_answer.flags
    assert bundle.task_meta.get("title") == "Introduction"

    # evidence is written
    ev = tmp_path / "evidence" / "T03.json"
    assert ev.exists()
    assert ev.read_text().strip() != ""


def test_consensus_chunk_and_flag(tmp_path: Path):
    plan_consensus = {
        "meta": {
            "agreement_threshold": 0.15,
            "latex_agreement_threshold": 0.1,
        },
        "blocks": {
            "T03": {
                "block_id": "T03",
                "text": "Consensus text from block",
                "block_type": "text",
                "flagged": True,
                "flag_reasons": ["High OCR Disagreement"],
                "agreement_score": 0.32,
                "latex_agreement_score": None,
                "text_backend": "nanonets-ocr2-3b",
            }
        },
    }

    task = {"id": "T03", "order": 3, "title": "Introduction", "consensus": {"block_id": "T03"}}
    indices = {
        "assignment": tmp_path / "missing_assignment",
        "assessment": tmp_path / "missing_assessment",
        "rubric": tmp_path / "missing_rubric",
        "user": tmp_path / "missing_user",
    }

    bundle = build_context_bundle(task, indices, evidence_dir=tmp_path / "evidence", plan_consensus=plan_consensus)

    assert bundle.user_answer.flags.get("ocr_uncertain") is True
    assert bundle.user_answer.chunks
    first = bundle.user_answer.chunks[0]
    assert first.source_name.startswith("consensus")
    assert first.text == "Consensus text from block"
