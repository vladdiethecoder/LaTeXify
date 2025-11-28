"""
Typed configuration package for LaTeXify.

This package exposes:
- ``Settings`` / ``settings``: environment-driven toggles (pydantic-settings).
- Typed helper enums: ``OCRBackendChoice``, ``MathOCRChoice``, ``VisionPresetChoice``.
- ``load_runtime_config``: loader for YAML-based pipeline/hardware/model config.
"""

from .settings import (
    Settings,
    settings,
    OCRBackendChoice,
    MathOCRChoice,
    VisionPresetChoice,
)
from .runtime import RuntimeConfig, load_runtime_config

__all__ = [
    "Settings",
    "settings",
    "OCRBackendChoice",
    "MathOCRChoice",
    "VisionPresetChoice",
    "RuntimeConfig",
    "load_runtime_config",
]

