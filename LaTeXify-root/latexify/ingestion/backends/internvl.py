from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, Tuple

from latexify.ingestion.backends.base import CouncilOutput, LayoutChunk
from latexify.utils.logging import log_warning

from .vision import VisionChatBackend
from ..internvl_hf_adapter import InternVLHFAdapter

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert academic document reader. Describe what the provided chunk contains "
    "in polished Markdown. Focus on clarity and accuracy."
)


HF_MODE = "hf"


class InternVLBackend(VisionChatBackend):
    def __init__(
        self,
        endpoint: str | None,
        model: str = "InternVL-3.5-14B",
        *,
        mode: str | None = None,
        hf_options: Dict[str, Any] | None = None,
    ) -> None:
        self.mode = (mode or os.environ.get("INTERNVL_MODE") or "vllm").strip().lower()
        if self.mode not in {HF_MODE, "vllm"}:
            self.mode = "vllm"
        self.hf_adapter = None
        if self.mode == HF_MODE:
            opts = hf_options.copy() if hf_options else {}
            self.hf_adapter = InternVLHFAdapter(
                runner_script=opts.get("runner_script"),
                model_dir=opts.get("model_dir"),
                python_executable=opts.get("python"),
                max_new_tokens=int(opts.get("max_new_tokens", 700)),
                extra_args=opts.get("extra_args"),
                device_map=opts.get("device_map"),
                max_memory=opts.get("max_memory"),
                cuda_visible_devices=opts.get("cuda_visible_devices"),
                offload_folder=opts.get("offload_folder"),
                timeout_seconds=opts.get("timeout_seconds"),
                env_overrides=opts.get("env_overrides"),
                retries=opts.get("retries", 1),
            )
        super().__init__(
            name="internvl",
            endpoint=endpoint,
            model=model,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            temperature=0.1,
            max_tokens=700,
        )

    async def process(self, chunk: LayoutChunk) -> CouncilOutput:
        if self.mode == HF_MODE and self.hf_adapter:
            return await self._process_hf(chunk)
        return await super().process(chunk)

    async def _process_hf(self, chunk: LayoutChunk) -> CouncilOutput:
        loop = asyncio.get_running_loop()
        prompt = self._build_runner_prompt(chunk)
        try:
            text, metadata = await loop.run_in_executor(
                None,
                self._run_hf_adapter,
                prompt,
                chunk,
            )
            return CouncilOutput(
                backend=self.name,
                chunk_id=chunk.chunk_id,
                page_index=chunk.page_index,
                text=text.strip(),
                confidence=0.8 if text.strip() else 0.4,
                metadata=metadata,
            )
        except Exception as exc:
            log_warning(
                "InternVL HF adapter failed",
                chunk_id=chunk.chunk_id,
                error=str(exc),
            )
            fallback = self._fallback_text(chunk)
            meta = {
                "warning": f"HF runner failed: {exc}",
                "runner": "hf",
            }
            return CouncilOutput(
                backend=self.name,
                chunk_id=chunk.chunk_id,
                page_index=chunk.page_index,
                text=fallback,
                confidence=0.35,
                metadata=meta,
            )

    def _run_hf_adapter(self, prompt: str, chunk: LayoutChunk) -> Tuple[str, Dict[str, Any]]:
        assert self.hf_adapter is not None
        return self.hf_adapter.generate(prompt, chunk.image_path)

    def _build_runner_prompt(self, chunk: LayoutChunk) -> str:
        parts = [
            f"Page {chunk.page_index + 1} chunk `{chunk.chunk_id}`.",
            "Describe the content faithfully, keeping math inline when possible.",
        ]
        snippet = (chunk.text or "").strip()
        if snippet:
            parts.append("Raw text excerpt:")
            parts.append(snippet[:2000])
        prompt = "\n".join(parts)
        if chunk.image_path:
            return f"<image>\n{prompt}"
        return prompt


__all__ = ["InternVLBackend"]
