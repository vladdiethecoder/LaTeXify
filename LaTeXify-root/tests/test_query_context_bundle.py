# tests/test_query_context_bundle.py
from __future__ import annotations
import os
from pathlib import Path
import pytest

pytest.importorskip("faiss")

from latexify.kb.query_index import build_context_bundle

def test_bundle_minimal(tmp_path: Path, monkeypatch):
    # Fast mode: no heavy model downloads
    monkeypatch.setenv("RAG_TEST_FAST", "1")

    # Create a tiny faux index dir with chunks + meta (no FAISS needed)
    run_dir = tmp_path / "kb" / "latex"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "chunks.jsonl").write_text(
        '{"id":"c1","text":"Use align with amsmath"}\n{"id":"c2","text":"Use cleveref for references"}\n',
        encoding="utf-8",
    )
    (run_dir / "faiss.meta.json").write_text(
        '{"metas":[{"id":"c1"},{"id":"c2"}]}',
        encoding="utf-8",
    )

    task = {"id": "t01", "title": "align equations", "anchor": "c1"}
    indices = {"assessment": str(run_dir), "assignment": str(run_dir), "rubric": str(run_dir), "user": str(run_dir)}
    bundle = build_context_bundle(task, indices, k_user=2, evidence_dir=tmp_path / "evidence")
    assert bundle["task_id"] == "t01"
    assert "question" in bundle and isinstance(bundle["question"], str)
    assert isinstance(bundle["rubric"], list) and len(bundle["rubric"]) > 0
    assert "chunks" in bundle["user_answer"]
