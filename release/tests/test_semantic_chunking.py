from collections import defaultdict

from release.pipeline import ingestion


class StubSemanticChunker:
    """Deterministic chunker used to simulate semantic breakpoints in tests."""

    def __init__(self, break_after: int) -> None:
        self.break_after = break_after
        self._seen = 0

    def sentence_count(self, text: str) -> int:
        return max(1, text.count("."))

    def embed(self, text: str):
        self._seen += 1
        return self._seen

    def should_break(self, prev_embedding, current_embedding, buffer_sentence_count: int) -> bool:
        if prev_embedding is None:
            return False
        return (
            buffer_sentence_count >= 1
            and prev_embedding < self.break_after <= current_embedding
        )


def test_semantic_chunking_breaks_without_char_limit():
    pages = [
        "Intro topic stays consistent.\n\nMore details on the same idea.\n\nNow a brand new topic emerges with different vocabulary.",
    ]
    chunker = StubSemanticChunker(break_after=2)
    chunks, _, _ = ingestion.chunk_text(
        pages=pages,
        page_images=defaultdict(list),
        chunk_chars=10_000,
        ocr_helper=None,
        page_store=None,
        semantic_chunker=chunker,
    )
    assert len(chunks) == 2
    assert "Intro topic" in chunks[0].text
    assert "brand new topic" in chunks[1].text
