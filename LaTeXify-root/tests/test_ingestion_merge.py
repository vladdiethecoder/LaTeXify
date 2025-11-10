from __future__ import annotations

from latexify.ingestion.orchestrator import CouncilOrchestrator


def _sample_outputs(better_text: str) -> list[dict]:
    return [
        {
            "backend": "internvl",
            "chunk_id": "chunk-1",
            "page_index": 0,
            "text": "placeholder",
            "confidence": 0.2,
            "metadata": {},
        },
        {
            "backend": "mineru",
            "chunk_id": "chunk-1",
            "page_index": 0,
            "text": better_text,
            "confidence": 0.5,
            "metadata": {},
        },
    ]


def test_merge_prefers_similar_text(monkeypatch, tmp_path):
    orchestrator = CouncilOrchestrator([], tmp_path)
    monkeypatch.setattr(orchestrator, "_support_scores", lambda outputs: [0.2, 0.95])
    outputs = _sample_outputs("Table | Value")
    merged = orchestrator._merge_chunk("chunk-1", outputs)
    assert merged["text"].startswith("Table")


def test_merge_uses_llm_fallback(monkeypatch, tmp_path):
    orchestrator = CouncilOrchestrator([], tmp_path)

    monkeypatch.setattr(orchestrator, "_llm_merge", lambda chunk_id, outputs: "LLM merged")
    monkeypatch.setattr(orchestrator, "_support_scores", lambda outputs: [0.1 for _ in outputs])
    outputs = [
        {
            "backend": "internvl",
            "chunk_id": "chunk-1",
            "page_index": 0,
            "text": "Option A",
            "confidence": 0.8,
            "metadata": {},
        },
        {
            "backend": "mineru",
            "chunk_id": "chunk-1",
            "page_index": 0,
            "text": "Completely different text B",
            "confidence": 0.8,
            "metadata": {},
        },
    ]
    merged = orchestrator._merge_chunk("chunk-1", outputs)
    assert merged["text"] == "LLM merged"
