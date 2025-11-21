"""Structured configuration for ingestion/backend toggles."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Literal, Mapping, MutableMapping, Tuple

from latexify.config import settings, VisionPresetChoice, OCRBackendChoice

MathOCRChoice = Literal["none", "pix2tex", "latex-ocr"]
BranchName = Literal["branch_a", "branch_b", "branch_c"]


@dataclass
class BackendToggleConfig:
    """Flags that describe which ingestion/math OCR backends should be active."""

    ocr_backend: OCRBackendChoice = "ensemble"
    layout_backend: str = "surya"
    surya_math_detector: bool = True
    mineru_enabled: bool = False
    marker_enabled: bool = False
    mcp_pdf_processor_enabled: bool = False
    math_ocr_backend: MathOCRChoice = "none"

    def resolve_ingestion_mode(self) -> str:
        """Map the high-level backend selection onto the ingestion OCR mode."""
        mapping = {
            "florence": "florence2",
            "ensemble": "auto",
            "mineru": "auto",  # placeholder until MinerU adapter lands
        }
        return mapping.get(self.ocr_backend, "auto")

    def as_dict(self) -> Dict[str, object]:
        """Materialize the toggle state for telemetry/manifest output."""
        return asdict(self)

    def wants_math_ocr(self) -> bool:
        return self.math_ocr_backend != "none"


def _env_flag(value: str | None, default: bool = True) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    return normalized not in {"0", "false", "off", "no"}


def _env_float(value: str | None, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _parse_int_list(value: str) -> tuple[int, ...]:
    tokens = [token.strip() for token in value.split(",")]
    return tuple(int(token) for token in tokens if token)


VISION_PRESETS: Dict[VisionPresetChoice, Dict[str, object]] = {
    "balanced": {},
    "fast": {
        "target_sizes": (192, 256),
        "max_views_per_chunk": 2,
        "enable_tensor_views": False,
        "brightness": 0.05,
        "contrast": 0.1,
        "saturation": 0.05,
    },
    "quality": {
        "target_sizes": (384, 448, 512), # Boosted for 32GB VRAM
        "max_views_per_chunk": 8,        # Increased views
        "pad_to_square": True,
        "brightness": 0.2,
        "contrast": 0.2,
        "saturation": 0.1,
    },
}


@dataclass
class VisionRuntimeConfig:
    """Runtime toggles for the optional vision synthesis stages."""

    enabled: bool = True
    preset: VisionPresetChoice = "quality" # Default to quality
    overrides: Dict[str, object] = field(default_factory=dict)

    def resolved_overrides(self) -> Dict[str, object]:
        preset_key = self.preset if self.preset in VISION_PRESETS else "quality"
        resolved = dict(VISION_PRESETS[preset_key])
        resolved.update(self.overrides)
        return resolved

    def as_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "enabled": self.enabled,
            "preset": self.preset,
        }
        if self.overrides:
            payload["overrides"] = self.overrides
        return payload


_VISION_ENV_PARSERS: Dict[str, tuple[str, Callable[[str], object]]] = {
    "TARGET_SIZES": ("target_sizes", _parse_int_list),
    "MAX_VIEWS": ("max_views_per_chunk", lambda v: _env_int(v, 4)),
    "PAD_TO_SQUARE": ("pad_to_square", lambda v: _env_flag(v, True)),
    "BRIGHTNESS": ("brightness", lambda v: _env_float(v, 0.1)),
    "CONTRAST": ("contrast", lambda v: _env_float(v, 0.15)),
    "SATURATION": ("saturation", lambda v: _env_float(v, 0.05)),
    "ENABLE_GRAYSCALE": ("enable_grayscale", lambda v: _env_flag(v, True)),
    "ENABLE_TENSOR_VIEWS": ("enable_tensor_views", lambda v: _env_flag(v, True)),
}


def build_vision_runtime_config(
    *,
    enabled: bool | None = None,
    preset: str | None = None,
    env: Mapping[str, str] | None = None,
) -> VisionRuntimeConfig:
    """Resolve CLI/environment preferences for the vision synthesis stages."""

    source = env or os.environ
    # Use settings default if env/arg not present
    default_enabled = settings.enable_vision_synthesis
    default_preset = settings.vision_preset

    resolved_enabled = enabled if enabled is not None else _env_flag(
        source.get("LATEXIFY_VISION_SYNTHESIS_ENABLED"), default_enabled
    )
    resolved_preset = (preset or source.get("LATEXIFY_VISION_SYNTHESIS_PRESET") or default_preset).lower()
    
    overrides: Dict[str, object] = {}
    for suffix, (field_name, parser) in _VISION_ENV_PARSERS.items():
        env_key = f"LATEXIFY_VISION_SYNTHESIS_{suffix}"
        if env_key in source:
            overrides[field_name] = parser(source[env_key])
    return VisionRuntimeConfig(enabled=resolved_enabled, preset=resolved_preset, overrides=overrides)


def build_backend_toggle_config(
    *,
    ocr_backend: str | None = None,
    layout_backend: str | None = None,
    surya_math_detector: bool | None = None,
    math_ocr_backend: str | None = None,
    mineru_enabled: bool | None = None,
    marker_enabled: bool | None = None,
    mcp_pdf_processor_enabled: bool | None = None,
    env: MutableMapping[str, str] | None = None,
) -> BackendToggleConfig:
    source: MutableMapping[str, str] = env if env is not None else os.environ
    
    # Defaults from settings
    resolved_ocr = (ocr_backend or source.get("LATEXIFY_OCR_BACKEND") or settings.ocr_backend).lower()
    resolved_layout = (layout_backend or source.get("LATEXIFY_LAYOUT_BACKEND") or settings.layout_backend).lower()
    
    resolved_math_flag = (
        surya_math_detector
        if surya_math_detector is not None
        else _env_flag(source.get("LATEXIFY_SURYA_MATH_DETECTOR"), True)
    )
    resolved_math_backend = (math_ocr_backend or source.get("LATEXIFY_MATH_OCR_BACKEND") or "none").lower()
    resolved_mineru = (
        mineru_enabled
        if mineru_enabled is not None
        else _env_flag(source.get("LATEXIFY_MINERU_ENABLED"), False)
    )
    resolved_marker = (
        marker_enabled
        if marker_enabled is not None
        else _env_flag(source.get("LATEXIFY_MARKER_BACKUP"), False)
    )
    resolved_mcp_pdf = (
        mcp_pdf_processor_enabled
        if mcp_pdf_processor_enabled is not None
        else _env_flag(source.get("LATEXIFY_ENABLE_MCP_PDF_PROCESSOR"), False)
    )
    config = BackendToggleConfig(
        ocr_backend=resolved_ocr,
        layout_backend=resolved_layout,
        surya_math_detector=resolved_math_flag,
        math_ocr_backend=resolved_math_backend,
        mineru_enabled=resolved_mineru,
        marker_enabled=resolved_marker,
        mcp_pdf_processor_enabled=resolved_mcp_pdf,
    )
    source["LATEXIFY_OCR_BACKEND"] = resolved_ocr
    source["LATEXIFY_LAYOUT_BACKEND"] = resolved_layout
    source["LATEXIFY_SURYA_MATH_DETECTOR"] = "1" if resolved_math_flag else "0"
    source["LATEXIFY_MATH_OCR_BACKEND"] = resolved_math_backend
    source["LATEXIFY_MINERU_ENABLED"] = "1" if resolved_mineru else "0"
    source["LATEXIFY_MARKER_BACKUP"] = "1" if resolved_marker else "0"
    source["LATEXIFY_ENABLE_MCP_PDF_PROCESSOR"] = "1" if resolved_mcp_pdf else "0"
    return config


@dataclass
class BranchRuntimeConfig:
    enabled: bool = False
    branches: Tuple[BranchName, ...] = ("branch_a", "branch_b", "branch_c")
    memory_limit_gb: float | None = None

    def as_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "enabled": self.enabled,
            "branches": list(self.branches),
        }
        if self.memory_limit_gb is not None:
            payload["memory_limit_gb"] = self.memory_limit_gb
        return payload


def _parse_branch_tokens(raw: str | None) -> Tuple[BranchName, ...]:
    if not raw:
        return ("branch_a", "branch_b", "branch_c")
    normalized: list[BranchName] = []
    mapping = {"a": "branch_a", "b": "branch_b", "c": "branch_c"}
    for token in raw.split(","):
        key = token.strip().lower()
        if not key:
            continue
        branch = mapping.get(key)
        if branch is None and key in mapping.values():
            branch = key  # type: ignore[assignment]
        if branch and branch not in normalized:
            normalized.append(branch)  # type: ignore[arg-type]
    return tuple(normalized) if normalized else ("branch_a", "branch_b", "branch_c")


def build_branch_runtime_config(
    *,
    enabled: bool | None = None,
    branches: str | None = None,
    memory_limit_gb: float | None = None,
    env: MutableMapping[str, str] | None = None,
) -> BranchRuntimeConfig:
    source: MutableMapping[str, str] = env if env is not None else os.environ
    resolved_enabled = (
        enabled if enabled is not None else _env_flag(source.get("LATEXIFY_ENABLE_MULTI_BRANCH"), False)
    )
    resolved_branches = _parse_branch_tokens(branches or source.get("LATEXIFY_BRANCHES"))
    resolved_memory = (
        memory_limit_gb
        if memory_limit_gb is not None
        else (
            _env_float(source.get("LATEXIFY_BRANCH_MEMORY_LIMIT"), float("nan"))
            if source.get("LATEXIFY_BRANCH_MEMORY_LIMIT")
            else None
        )
    )
    source["LATEXIFY_ENABLE_MULTI_BRANCH"] = "1" if resolved_enabled else "0"
    source["LATEXIFY_BRANCHES"] = ",".join(resolved_branches)
    if resolved_memory is not None:
        source["LATEXIFY_BRANCH_MEMORY_LIMIT"] = str(resolved_memory)
    elif "LATEXIFY_BRANCH_MEMORY_LIMIT" in source:
        source.pop("LATEXIFY_BRANCH_MEMORY_LIMIT")
    return BranchRuntimeConfig(
        enabled=resolved_enabled,
        branches=resolved_branches,
        memory_limit_gb=resolved_memory,
    )


@dataclass
class KimiRuntimeConfig:
    temperature: float = 0.05
    context_size: int = 32768


@dataclass
class CompilationRuntimeConfig:
    enable_robust_compilation: bool = True
    retry_count: int = 3
    layout_confidence_threshold: float = 0.0
    monkey_ocr_enabled: bool = True


def build_kimi_runtime_config(
    *,
    temperature: float | None = None,
    context_size: int | None = None,
    env: MutableMapping[str, str] | None = None,
) -> KimiRuntimeConfig:
    source: MutableMapping[str, str] = env if env is not None else os.environ
    resolved_temp = temperature if temperature is not None else _env_float(
        source.get("LATEXIFY_KIMI_K2_TEMPERATURE"),
        0.05,
    )
    resolved_ctx = context_size if context_size is not None else _env_int(
        source.get("LATEXIFY_KIMI_K2_CONTEXT"),
        32768,
    )
    source["LATEXIFY_KIMI_K2_TEMPERATURE"] = str(resolved_temp)
    source["LATEXIFY_KIMI_K2_CONTEXT"] = str(resolved_ctx)
    return KimiRuntimeConfig(temperature=resolved_temp, context_size=resolved_ctx)


def build_compilation_runtime_config(
    *,
    enable_robust_compilation: bool | None = None,
    retry_count: int | None = None,
    layout_confidence_threshold: float | None = None,
    monkey_ocr_enabled: bool | None = None,
    env: MutableMapping[str, str] | None = None,
) -> CompilationRuntimeConfig:
    source: MutableMapping[str, str] = env if env is not None else os.environ
    resolved_enable = (
        enable_robust_compilation
        if enable_robust_compilation is not None
        else _env_flag(source.get("LATEXIFY_ENABLE_ROBUST_COMPILATION"), settings.enable_robust_compilation)
    )
    resolved_retry = retry_count if retry_count is not None else _env_int(
        source.get("LATEXIFY_COMPILATION_RETRY_COUNT"),
        settings.compilation_retry_count,
    )
    resolved_layout = (
        layout_confidence_threshold
        if layout_confidence_threshold is not None
        else _env_float(source.get("LATEXIFY_LAYOUT_CONFIDENCE_THRESHOLD"), settings.layout_confidence_threshold)
    )
    resolved_monkey = (
        monkey_ocr_enabled
        if monkey_ocr_enabled is not None
        else _env_flag(source.get("LATEXIFY_ENABLE_MONKEY_OCR"), settings.enable_monkey_ocr)
    )
    source["LATEXIFY_ENABLE_ROBUST_COMPILATION"] = "1" if resolved_enable else "0"
    source["LATEXIFY_COMPILATION_RETRY_COUNT"] = str(resolved_retry)
    source["LATEXIFY_LAYOUT_CONFIDENCE_THRESHOLD"] = str(resolved_layout)
    source["LATEXIFY_ENABLE_MONKEY_OCR"] = "1" if resolved_monkey else "0"
    return CompilationRuntimeConfig(
        enable_robust_compilation=resolved_enable,
        retry_count=resolved_retry,
        layout_confidence_threshold=resolved_layout,
        monkey_ocr_enabled=resolved_monkey,
    )


__all__ = [
    "BackendToggleConfig",
    "build_backend_toggle_config",
    "VisionRuntimeConfig",
    "VISION_PRESETS",
    "build_vision_runtime_config",
    "KimiRuntimeConfig",
    "CompilationRuntimeConfig",
    "build_kimi_runtime_config",
    "build_compilation_runtime_config",
]