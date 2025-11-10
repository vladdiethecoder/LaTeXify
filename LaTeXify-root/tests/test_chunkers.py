from __future__ import annotations

from pathlib import Path

from latexify.ingestion.chunkers import fixed_chunk_pages, semantic_chunk_pages
from latexify.ingestion.orchestrator import CouncilOrchestrator


def test_semantic_chunk_pages_assigns_images(tmp_path):
    images = {0: [tmp_path / "page0001-img01.png"]}
    pages = ["Paragraph one.\n\nParagraph two."]
    chunks = semantic_chunk_pages(pages, chunk_chars=10, images_by_page=images)
    assert len(chunks) >= 1
    assert chunks[0].image_path == images[0][0]
    assert "paragraph" in chunks[0].text.lower()


def test_semantic_chunk_detects_headings():
    pages = ["SECTION 1: Intro\nDetails\n\nFigure 1: Example\nCaption text"]
    chunks = semantic_chunk_pages(pages, chunk_chars=80)
    assert any("SECTION 1" in chunk.text for chunk in chunks)
    assert any("Figure 1" in chunk.text for chunk in chunks)


def test_fixed_chunk_pages_assigns_images(tmp_path):
    images = {0: [tmp_path / "page0001.png", tmp_path / "page0001b.png"]}
    pages = ["A" * 30]
    chunks = fixed_chunk_pages(pages, chunk_chars=10, images_by_page=images)
    assert len(chunks) >= 2
    assert chunks[1].image_path == images[0][1]


def test_council_merge_prefers_meaningful_output(tmp_path):
    asset_path = tmp_path / "figure.png"
    orchestrator = CouncilOrchestrator([], tmp_path, chunk_assets={"chunk-1": [str(asset_path)]})
    outputs = [
        {"backend": "internvl", "chunk_id": "chunk-1", "page_index": 0, "text": "", "confidence": 0.9, "metadata": {}},
        {
            "backend": "mineru",
            "chunk_id": "chunk-1",
            "page_index": 0,
            "text": "Table content",
            "confidence": 0.2,
            "metadata": {},
        },
    ]
    merged = orchestrator._merge_chunk("chunk-1", outputs)
    assert merged["text"] == "Table content"
    assert merged["assets"] == [str(asset_path)]
