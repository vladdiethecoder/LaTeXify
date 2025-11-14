from collections import OrderedDict, defaultdict
from pathlib import Path

from release.pipeline import ingestion


def test_merge_text_sources_deduplicates_paragraphs():
    merged = ingestion.merge_text_sources(
        [
            "Alpha paragraph.\n\nBeta paragraph.",
            "Beta paragraph.\n\nGamma paragraph.",
        ]
    )
    assert "Alpha paragraph." in merged
    assert "Gamma paragraph." in merged
    assert merged.count("Beta paragraph.") == 1


class StubOCR:
    def extract(self, page_index: int):
        return ingestion.OCRResult(
            OrderedDict(
                [
                    ("florence2", "Diagram overview.\n\nShared paragraph."),
                    ("internvl", "Shared paragraph.\n\nEquation block."),
                ]
            )
        )


class NoopChunker:
    def sentence_count(self, text: str) -> int:
        return 1

    def embed(self, text: str):
        return None

    def should_break(self, *args, **kwargs) -> bool:
        return False


def test_chunk_text_merges_all_backends():
    pages = ["Base text paragraph.\n\nShared paragraph."]
    chunks, usage, _ = ingestion.chunk_text(
        pages=pages,
        page_images=defaultdict(list),
        chunk_chars=10_000,
        ocr_helper=StubOCR(),
        page_store=None,
        semantic_chunker=NoopChunker(),
    )
    assert chunks, "expected at least one chunk"
    chunk = chunks[0]
    assert "Base text paragraph." in chunk.text
    assert "Diagram overview." in chunk.text
    assert "Equation block." in chunk.text
    assert chunk.metadata["ocr_backends"] == ["pypdf", "florence2", "internvl"]
    assert usage["pypdf"] == 1
    assert usage["florence2"] == 1
    assert usage["internvl"] == 1


def test_auto_download_missing_models(monkeypatch, tmp_path):
    calls = []

    def fake_snapshot(repo_id, local_dir, **kwargs):
        calls.append(repo_id)
        path = Path(local_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(ingestion, "snapshot_download", fake_snapshot)
    fallback = ingestion.OCRFallback("auto", tmp_path, None, tmp_path)
    target = tmp_path / "ocr" / "nougat-small"
    assert not target.exists()
    fallback._ensure_backend_weights("nougat")
    assert target.exists()
    fallback._ensure_backend_weights("nougat")  # second call should be no-op
    assert len(calls) == 1
