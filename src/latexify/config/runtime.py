from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from omegaconf import OmegaConf
from pydantic import BaseModel, Field


class IngestionConfig(BaseModel):
    backend: str = Field(default="docling")
    dpi: int = Field(default=300)
    chunk_chars: int = Field(default=2000)
    docling: dict[str, Any] = Field(default_factory=dict)


class RefinementConfig(BaseModel):
    enabled: bool = Field(default=True)
    use_vllm: bool = Field(default=True)
    load_in_8bit: bool = Field(default=False)
    load_in_4bit: bool = Field(default=False)
    refinement_passes: int = Field(default=1)
    llm_repo: Optional[str] = Field(default=None)


class HardwareConfig(BaseModel):
    llm_device: str = Field(default="cuda")


class PipelineConfig(BaseModel):
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    refinement: RefinementConfig = Field(default_factory=RefinementConfig)


class RuntimeConfig(BaseModel):
    """Typed view over the merged pipeline + hardware configuration."""

    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)


def load_runtime_config(config_dir: Path | str = "config") -> RuntimeConfig:
    """Load and normalize YAML pipeline + hardware configuration.

    This function replaces ad-hoc OmegaConf usage in entrypoints and provides
    a stable, typed object for the core pipeline.
    """
    base_dir = Path(config_dir)
    pipeline_cfg = OmegaConf.load(base_dir / "pipeline.yaml")

    # Optional hardware/model overlays.
    hardware_path = base_dir / "hardware" / "rtx5090.yaml"
    hw_cfg = OmegaConf.load(hardware_path) if hardware_path.exists() else OmegaConf.create({})

    merged = OmegaConf.merge(
        OmegaConf.create({"hardware": hw_cfg}),
        pipeline_cfg,
    )
    container = OmegaConf.to_container(merged, resolve=True)  # type: ignore[arg-type]

    pipeline_data = container.get("pipeline", {}) if isinstance(container, dict) else {}
    hardware_data = container.get("hardware", {}) if isinstance(container, dict) else {}

    return RuntimeConfig(
        pipeline=PipelineConfig(
            ingestion=IngestionConfig(**pipeline_data.get("ingestion", {})),
            refinement=RefinementConfig(**pipeline_data.get("refinement", {})),
        ),
        hardware=HardwareConfig(**hardware_data),
    )


__all__ = ["RuntimeConfig", "load_runtime_config"]

