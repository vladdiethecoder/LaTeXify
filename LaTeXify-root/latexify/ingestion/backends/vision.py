from __future__ import annotations

import asyncio
import base64
import json
import os
import textwrap
import urllib.request
from typing import Any, Dict, List, MutableMapping

from latexify.utils.logging import log_warning

from .base import BaseCouncilBackend, CouncilOutput, LayoutChunk


def _encode_image(path):
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def _normalize_endpoint(base: str) -> str:
    base = (base or "").strip()
    if not base:
        return ""
    if base.rstrip("/").endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return base.rstrip("/") + "/chat/completions"
    return base.rstrip("/") + "/v1/chat/completions"


class VisionChatBackend(BaseCouncilBackend):
    def __init__(
        self,
        *,
        name: str,
        endpoint: str | None,
        model: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> None:
        self.name = name
        self.endpoint = _normalize_endpoint(endpoint or "")
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def process(self, chunk: LayoutChunk) -> CouncilOutput:
        if self.endpoint:
            try:
                text, meta = await asyncio.get_running_loop().run_in_executor(
                    None, self._call_endpoint, chunk
                )
                return CouncilOutput(
                    backend=self.name,
                    chunk_id=chunk.chunk_id,
                    page_index=chunk.page_index,
                    text=text.strip(),
                    confidence=meta.get("confidence", 0.78),
                    metadata=meta,
                )
            except Exception as exc:
                log_warning(
                    "Vision backend request failed",
                    backend=self.name,
                    chunk_id=chunk.chunk_id,
                    error=str(exc),
                )
                meta = {"warning": f"{self.name} request failed: {exc}"}
        else:
            meta = {"warning": f"{self.name} endpoint not configured; using fallback"}
        fallback = self._fallback_text(chunk)
        return CouncilOutput(
            backend=self.name,
            chunk_id=chunk.chunk_id,
            page_index=chunk.page_index,
            text=fallback,
            confidence=0.35,
            metadata=meta,
        )

    def _build_messages(self, chunk: LayoutChunk) -> List[Dict[str, Any]]:
        user_parts: List[Dict[str, Any]] = []

        def add_text(txt: str) -> None:
            user_parts.append({"type": "text", "text": txt})

        add_text(
            "\n".join(
                [
                    f"Page {chunk.page_index + 1} chunk `{chunk.chunk_id}`.",
                    "Describe the text faithfully and keep math inline when possible.",
                ]
            )
        )
        if chunk.image_path and chunk.image_path.exists():
            add_text("[Attached chunk image]")
            user_parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _encode_image(chunk.image_path), "detail": "high"},
                }
            )
        if chunk.text:
            add_text("Raw text:")
            add_text(chunk.text[:4000])
        return [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": user_parts}]

    def _call_endpoint(self, chunk: LayoutChunk) -> tuple[str, Dict[str, Any]]:
        messages = self._build_messages(chunk)
        headers: MutableMapping[str, str] = {"Content-Type": "application/json"}
        api_key = os.environ.get("LATEXIFY_COUNCIL_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        req = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310
            body = resp.read().decode("utf-8")
        parsed = json.loads(body)
        choices = parsed.get("choices") or []
        if not choices:
            raise RuntimeError(f"{self.name} returned no choices")
        message = choices[0].get("message") or {}
        text = message.get("content", "")
        meta = {
            "finish_reason": choices[0].get("finish_reason"),
            "usage": parsed.get("usage"),
            "raw_response": text,
        }
        cleaned = self._extract_text(text)
        return cleaned, meta

    def _extract_text(self, payload: str) -> str:
        return payload.strip()

    def _fallback_text(self, chunk: LayoutChunk) -> str:
        lines = textwrap.wrap(chunk.text or "", width=80)
        snippet = "\n".join(lines[:6]).strip()
        return snippet or "(no text extracted)"


__all__ = ["VisionChatBackend"]
