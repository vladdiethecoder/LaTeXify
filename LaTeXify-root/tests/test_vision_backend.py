from __future__ import annotations

import asyncio
import json

from latexify.ingestion.backends.base import LayoutChunk
from latexify.ingestion.backends.vision import VisionChatBackend


class DummyBackend(VisionChatBackend):
    def __init__(self, endpoint: str) -> None:
        super().__init__(
            name="dummy",
            endpoint=endpoint,
            model="dummy-model",
            system_prompt="You are helpful",
            temperature=0.0,
            max_tokens=32,
        )


class DummyResponse:
    def __init__(self, body: str) -> None:
        self._body = body.encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "DummyResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:  # pragma: no cover - nothing special
        return False


def test_vision_backend_calls_endpoint(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout):  # noqa: ANN001
        captured["url"] = req.full_url
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        body = json.dumps(
            {
                "choices": [
                    {
                        "message": {"content": "Recognized text."},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }
        )
        return DummyResponse(body)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    backend = DummyBackend("https://example.com/v1")
    chunk = LayoutChunk(
        chunk_id="chunk-1",
        page_index=0,
        text="sample text",
        page_name="page-0001.md",
    )
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(backend.process(chunk))
    finally:  # pragma: no cover - best effort cleanup
        loop.close()
    assert result.text == "Recognized text."
    assert captured["payload"]["model"] == "dummy-model"
    assert captured["payload"]["messages"], "messages payload missing"
