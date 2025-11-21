"""Flux/SDXL-style diffusion wrapper for render-aware reconstruction."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict

from PIL import Image

try:  # pragma: no cover - heavy dependency
    from diffusers import DiffusionPipeline
    import torch
except Exception:  # pragma: no cover
    DiffusionPipeline = None  # type: ignore
    torch = None  # type: ignore

LOGGER = logging.getLogger(__name__)


@dataclass
class FluxConfig:
    model_id: str = "black-forest-labs/Flux.1-Fill-dev"
    controlnet_id: str | None = None
    dtype: str = "fp16"
    device: str = "auto"
    steps: int = 20
    guidance: float = 1.0
    prompt: str = (
        "High-quality academic page, clean serif typography, precise equations,"
        " Springer style lighting, subtle paper texture."
    )


class FluxInpaintingEngine:
    """Wraps a Diffusers inpainting pipeline configured for Flux Fill models."""

    def __init__(
        self,
        config: FluxConfig,
        *,
        workdir: Path,
        pipeline_loader: Callable[[FluxConfig], Any] | None = None,
    ) -> None:
        self.config = config
        self.workdir = workdir
        self.workdir.mkdir(parents=True, exist_ok=True)
        self._pipeline_loader = pipeline_loader or self._default_loader
        self._pipeline = None

    def _default_loader(self, cfg: FluxConfig):  # pragma: no cover - heavy dep
        if DiffusionPipeline is None or torch is None:
            raise ImportError(
                "diffusers[torch] is required to run render-aware diffusion. Install diffusers and torch."
            )
        LOGGER.info("Loading diffusion pipeline %s", cfg.model_id)
        dtype = torch.float16 if cfg.dtype == "fp16" and torch.cuda.is_available() else torch.float32
        pipe = DiffusionPipeline.from_pretrained(cfg.model_id, torch_dtype=dtype)
        device = self._resolve_device(cfg.device)
        pipe.to(device)
        return pipe

    @staticmethod
    def _resolve_device(device_pref: str) -> str:
        if device_pref == "auto":
            if torch is not None and torch.cuda.is_available():
                return "cuda"
            return "cpu"
        return device_pref

    def _ensure_pipeline(self):
        if self._pipeline is None:
            self._pipeline = self._pipeline_loader(self.config)
        return self._pipeline

    def generate_page(
        self,
        constraint_map: Path,
        mask_path: Path,
        *,
        prompt: str | None = None,
        steps: int | None = None,
        guidance: float | None = None,
        page_index: int,
    ) -> Path:
        pipeline = self._ensure_pipeline()
        base_prompt = prompt or self.config.prompt
        num_steps = steps or self.config.steps
        guidance_scale = guidance or self.config.guidance
        init = Image.open(constraint_map).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        LOGGER.info(
            "Running flux diffusion for page %s (steps=%s, guidance=%.2f)",
            page_index,
            num_steps,
            guidance_scale,
        )
        if hasattr(pipeline, "__call__"):
            output = pipeline(
                prompt=base_prompt,
                image=init,
                mask_image=mask,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
            )
            if hasattr(output, "images") and output.images:
                result = output.images[0]
            else:  # pragma: no cover - defensive
                result = init
        else:  # pragma: no cover - defensive
            LOGGER.warning("Diffusion pipeline missing __call__; returning constraint map.")
            result = init
        target = self.workdir / f"page_{page_index:04d}_render.png"
        result.save(target)
        return target


__all__ = ["FluxInpaintingEngine", "FluxConfig"]
