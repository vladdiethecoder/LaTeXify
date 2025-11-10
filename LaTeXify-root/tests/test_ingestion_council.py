from __future__ import annotations

import json
from pathlib import Path

from latexify.ingestion.chunkers import semantic_chunk_pages
from latexify.ingestion.ingest_pdf import ingest_pdf


def test_chunk_pages_simple():
    pages = ["Header\n\nParagraph one.\n\nParagraph two with more text."]
    chunks = semantic_chunk_pages(pages, chunk_chars=40, min_chars=5)
    assert len(chunks) >= 1
    assert chunks[0].chunk_id.startswith("page0001")
    assert chunks[0].text


def test_ingest_pdf_offline(tmp_path):
    run_dir = tmp_path / "run"
    pdf_stub = tmp_path / "stub.pdf"
    pdf_stub.write_text("placeholder", encoding="utf-8")
    pages = [
        "CHAPTER 1: Systems\n\nThis page references Figure 1 and Table 1.\nvalue  count\nA      10\nB      12\nEquation y = mx + b.",
        "Section 2 Notes\n\nAnother paragraph with no math.",
    ]
    summary = ingest_pdf(
        pdf=pdf_stub,
        run_dir=run_dir,
        pages_override=pages,
    )
    assert summary["chunk_count"] >= 1
    assert summary["consensus_dir"]
    manifest = json.loads((run_dir / "council" / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["records"]) >= summary["chunk_count"] * 4
    sample = run_dir / "outputs" / "mineru"
    assert any(sample.iterdir()), "mineru outputs missing"
    consensus_files = [p for p in (run_dir / "consensus").glob("*.json") if p.name != "manifest.json"]
    assert consensus_files, "consensus snippets missing"
    record = json.loads(consensus_files[0].read_text(encoding="utf-8"))
    assert record["content_type"] in {"text", "math", "table", "code", "chem", "figure_with_text"}


def test_ingest_pdf_resilience_report(tmp_path):
    run_dir = tmp_path / "run"
    pdf_stub = tmp_path / "stub.pdf"
    pdf_stub.write_text("placeholder", encoding="utf-8")
    pages = ["", ""]
    summary = ingest_pdf(
        pdf=pdf_stub,
        run_dir=run_dir,
        pages_override=pages,
        permissive=True,
    )
    report_path = run_dir / "resilience_report.json"
    assert summary["resilience_report"] == str(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["permissive_backends"] == ["generic_ocr"]
    assert "stats" in report and isinstance(report["stats"], dict)
