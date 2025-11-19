"""Vision-stage utilities (multi-view rendering + agents)."""

from .synthesis import MultiViewRenderer, VisionSynthesisConfig, VisionView, VisionSynthesisResult
from .agents import (
    BaseVisionAgent,
    VisionAgentResult,
    VisionAgentSuite,
    MathematicalVisionAgent,
    SemanticVisionAgent,
    StructuralVisionAgent,
    StyleVisionAgent,
)

__all__ = [
    "MultiViewRenderer",
    "VisionSynthesisConfig",
    "VisionView",
    "VisionSynthesisResult",
    "BaseVisionAgent",
    "VisionAgentResult",
    "VisionAgentSuite",
    "MathematicalVisionAgent",
    "SemanticVisionAgent",
    "StructuralVisionAgent",
    "StyleVisionAgent",
]
