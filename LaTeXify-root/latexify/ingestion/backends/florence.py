from __future__ import annotations

import json

from latexify.utils.logging import log_warning

from .vision import VisionChatBackend

DEFAULT_FLORENCE_PROMPT = (
    "You are Florence-2 running in OCR-with-region mode. Return JSON with an array `items`, "
    "each containing `text` and normalized bounding box coordinates [x0,y0,x1,y1]."
)


class FlorenceBackend(VisionChatBackend):
    def __init__(self, endpoint: str | None, model: str = "Florence-2") -> None:
        super().__init__(
            name="florence2",
            endpoint=endpoint,
            model=model,
            system_prompt=DEFAULT_FLORENCE_PROMPT,
            temperature=0.0,
            max_tokens=700,
        )

    def _extract_text(self, payload: str) -> str:
        try:
            data = json.loads(payload)
            items = data.get("items") or []
            lines = []
            for item in items:
                bbox = item.get("bbox") or [0, 0, 1, 1]
                lines.append(f"[{bbox}] {item.get('text','').strip()}")
            return "\n".join(lines)
        except Exception as exc:
            log_warning("Florence backend payload parse failed", error=str(exc))
            return payload.strip()

    def _fallback_text(self, chunk):
        norm_bbox = chunk.bbox or [0.0, 0.0, 1.0, 1.0]
        return json.dumps(
            {"items": [{"text": (chunk.text or "").strip(), "bbox": norm_bbox, "source": "heuristic"}]},
            ensure_ascii=False,
        )


__all__ = ["FlorenceBackend"]
