import pytest

from latexify.pipeline import semantic_chunking


class BoomEncoder:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("missing model")


def test_semantic_chunker_errors_when_transformer_missing(monkeypatch):
    monkeypatch.setattr(semantic_chunking, "_TransformerEncoder", BoomEncoder)
    with pytest.raises(semantic_chunking.SemanticChunkerError):
        semantic_chunking.SemanticChunker(allow_hash_fallback=False, encoder_backend="auto")


def test_semantic_chunker_falls_back_when_allowed(monkeypatch):
    monkeypatch.setattr(semantic_chunking, "_TransformerEncoder", BoomEncoder)
    chunker = semantic_chunking.SemanticChunker(allow_hash_fallback=True, encoder_backend="auto")
    assert isinstance(chunker.encoder, semantic_chunking._HashingEncoder)
