"""
Base abstractions for the LaTeXify Gen 3.0 agent system.

This module defines the core interfaces that all extraction agents must implement,
following the Strategy Pattern for swappable backends.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
from pydantic import BaseModel, Field
from PIL import Image


class AgentResult(BaseModel):
    """Normalized result returned by agents.

    Attributes:
        content: Primary textual content (typically LaTeX or plain text).
        confidence: Optional confidence score between 0.0 and 1.0.
        metadata: Arbitrary structured metadata (backend name, tags, etc.).
    """

    content: str = Field(...)
    confidence: Optional[float] = Field(default=None)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseExtractor(ABC):
    """Abstract base class for all content extraction backends.

    Extraction backends process individual crops (text, math, tables, figures)
    and return raw strings that higher-level agents post-process.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the extractor with optional configuration."""
        self.config: Dict[str, Any] = config or {}
        self._initialized: bool = False

    @abstractmethod
    def extract(self, image: np.ndarray) -> str:
        """Extract content from a single image crop."""

    def extract_batch(self, images: list[np.ndarray]) -> list[str]:
        """Extract content from multiple crops (default serial implementation)."""
        return [self.extract(img) for img in images]

    def get_confidence(self) -> Optional[float]:
        """Return confidence score of the last extraction, if available."""
        return None

    def warmup(self) -> None:
        """Perform model warmup (load weights, compile, etc.)."""
        if not self._initialized:
            self._initialize()
            self._initialized = True

    @abstractmethod
    def _initialize(self) -> None:
        """Internal initialization logic (load model weights, etc.)."""


class BaseAgent(ABC):
    """Higher-level agent that orchestrates extractors and post-processing.

    Agents wrap one or more `BaseExtractor` instances and add domain-specific
    logic such as validation, routing, and error handling.
    """

    def __init__(self, extractors: Dict[str, BaseExtractor]) -> None:
        """Initialize agent with a dictionary of named extractors."""
        self.extractors: Dict[str, BaseExtractor] = extractors

    @abstractmethod
    def process(
        self,
        image: np.ndarray,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Process a single crop and return a legacy result dictionary."""

    def run(
        self,
        image: np.ndarray,
        context: Optional[Dict[str, Any]] = None,
    ) -> AgentResult:
        """Process an image crop and return a normalized `AgentResult`.

        This method provides a typed façade over the historical `process`
        API. New code should prefer `run()`, while `process()` remains for
        backwards compatibility.
        """
        raw = self.process(image, context)
        return AgentResult(
            content=str(raw.get("content", "")),
            confidence=raw.get("confidence"),
            metadata=dict(raw.get("metadata") or {}),
        )
