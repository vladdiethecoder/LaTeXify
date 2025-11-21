"""Utilities for resolving model directories with local fallbacks."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List

_REQUIRED_SUBDIRS: List[Path] = [Path("ocr") / "mineru-1.2b"]
_STATIC_FALLBACKS: List[Path] = [
    Path("/run/media/vdubrov/Active Storage/models"),
]
LOGGER = logging.getLogger(__name__)

DEFAULT_MATH_MODEL = "Qwen/Qwen2.5-Math-7B-Instruct"
DEFAULT_VISION_MODEL = "Qwen/Qwen2-VL-7B-Instruct"


def _parse_extra_paths(raw: str | None) -> List[Path]:
    if not raw:
        return []
    return [Path(token).expanduser() for token in raw.split(os.pathsep) if token.strip()]


def _has_required_payload(base: Path) -> bool:
    for subdir in _REQUIRED_SUBDIRS:
        if not (base / subdir).exists():
            return False
    return True


def resolve_models_root(default_path: Path) -> Path:
    """Resolve the models directory with support for external fallbacks.

    Priority order:
    1. LATEXIFY_MODELS_ROOT (if set and exists)
    2. default_path (repo-local)
    3. LATEXIFY_MODELS_FALLBACKS entries (os.pathsep separated)
    4. Static mountpoints such as /run/media/vdubrov/Active Storage/models

    When a fallback contains required payloads (e.g., MinerU weights) that the primary
    path lacks, it becomes the preferred directory.
    """

    candidates: List[Path] = []
    env_path = os.environ.get("LATEXIFY_MODELS_ROOT")
    if env_path:
        candidates.append(Path(env_path).expanduser())

    candidates.append(default_path.expanduser())

    for extra in _parse_extra_paths(os.environ.get("LATEXIFY_MODELS_FALLBACKS")):
        if extra not in candidates:
            candidates.append(extra)

    for fallback in _STATIC_FALLBACKS:
        if fallback not in candidates:
            candidates.append(fallback)

    preferred: Path | None = None
    payload_match: Path | None = None
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except FileNotFoundError:
            continue
        if not resolved.exists():
            continue
        if _has_required_payload(resolved):
            payload_match = resolved
            break
        if preferred is None:
            preferred = resolved
    chosen = payload_match or preferred or default_path.resolve()
    if payload_match:
        LOGGER.info("Using models root %s (contains MinerU weights).", payload_match)
    elif preferred and preferred != default_path.resolve():
        LOGGER.info("Using fallback models root %s.", preferred)
    return chosen
