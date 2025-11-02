# tests/test_retrieval_bundle_fallback.py
from __future__ import annotations
import json
from pathlib import Path
import importlib

def test_fallback_uses_corpus_head_when_no_deps(tmp_path: Path, monkeypatch):
    # Create a minimal "corpus" under kb/latex/chunks.jsonl
    run_dir = tmp_path / "kb/latex"
    run_dir.mkdir(parents=True)
    rows = [
        {"id":"a1","text":"First chunk text."},
        {"id":"a2","text":"Second chunk text."},
        {"id":"a3","text":"Third chunk text."},
    ]
    (run_dir / "chunks.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    # Import module and force flags to simulate missing deps
    import latexify.pipeline.retrieval_bundle as rb
    importlib.reload(rb)
    monkeypatch.setattr(rb, "_FAISS_OK", False, raising=False)
    monkeypatch.setattr(rb, "_SBERT_OK", False, raising=False)

    # Build a tiny plan and call builder
    task = {"id": "T03", "title": "Introduction"}
    indices = {"assignment": run_dir, "assessment": run_dir, "rubric": run_dir, "user": run_dir}
    bundle = rb.build_context_bundle(task, indices, evidence_dir=tmp_path / "evidence")

    # Expect chunks via fallback (from head of sorted ids)
    assert len(bundle.user_answer.chunks) > 0
    ids = [c.id for c in bundle.user_answer.chunks]
    assert ids[0] == "a1"
