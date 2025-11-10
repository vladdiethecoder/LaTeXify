from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence


@dataclass
class LayoutChunk:
    chunk_id: str
    page_index: int
    text: str
    page_name: str
    bbox: List[float] | None = None
    image_path: Path | None = None


@dataclass
class CouncilOutput:
    backend: str
    chunk_id: str
    page_index: int
    text: str
    confidence: float
    metadata: Dict[str, Any]


class BaseCouncilBackend:
    """Abstract council backend."""

    name: str = "base"

    async def process(self, chunk: LayoutChunk) -> CouncilOutput:
        raise NotImplementedError


__all__ = ["LayoutChunk", "CouncilOutput", "BaseCouncilBackend"]
