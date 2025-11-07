from __future__ import annotations

import json
from pathlib import Path

from latexify.pipeline import langchain_orchestrator as orchestrator


def _make_dummy_pdf(tmp_path: Path) -> Path:
    pdf = tmp_path / "dummy.pdf"
    pdf.write_text("placeholder", encoding="utf-8")
    return pdf


def test_orchestrator_offline_chain(tmp_path):
    pdf = _make_dummy_pdf(tmp_path)
    cfg = orchestrator.PipelineConfig(
        pdf=pdf,
        title="Test Doc",
        author="Unit Tester",
        build_dir=tmp_path / "build",
        run_root=tmp_path / "runs",
        pages_override=["Header\n\nParagraph one", "Table | Value"],
        aggregate=False,
    )
    state = orchestrator.run_pipeline(cfg, use_langchain=False)
    assert state.run_dir and state.run_dir.exists()
    assert state.plan_path and state.plan_path.exists()
    assert state.consensus_path and state.consensus_path.exists()
    assert state.snippets_dir and state.snippets_dir.exists()
    assert state.qa_report and state.qa_report.exists()
    report = json.loads(state.qa_report.read_text(encoding="utf-8"))
    assert report["snippets_checked"] >= 1
