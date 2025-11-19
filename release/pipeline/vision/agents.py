"""Vision-aware agent scaffolding that consumes multi-view crops."""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence

from ...core import common
from ...models.vllm_client import get_vllm_client
from ..specialists import VisionLanguageGenerator
from .synthesis import VisionView


DEFAULT_VISION_AGENT_VLM = os.environ.get("LATEXIFY_VISION_AGENT_VLM", "internvl")


@dataclass
class VisionAgentResult:
    chunk_id: str
    agent: str
    confidence: float
    summary: str
    metadata: Dict[str, object] = field(default_factory=dict)


class BaseVisionAgent(ABC):
    """Shared utilities for specialized multi-view reasoning agents."""

    def __init__(
        self,
        name: str,
        prompt: str,
        *,
        vlm_backend: str = DEFAULT_VISION_AGENT_VLM,
        max_tokens: int = 192,
    ) -> None:
        self.name = name
        self._prompt_template = prompt
        self._vlm = VisionLanguageGenerator(prompt, vlm_backend, max_new_tokens=max_tokens)
        self._llm_client = get_vllm_client()

    def analyze(self, chunk: common.Chunk, views: Sequence[VisionView]) -> VisionAgentResult:
        descriptions = self._describe_views(views)
        summary = self._summarize(chunk, descriptions)
        confidence = self._score(chunk, descriptions)
        metadata = {
            "view_count": len(views),
            "descriptions": descriptions,
        }
        metadata.update(self._extra_metadata(chunk, descriptions))
        return VisionAgentResult(
            chunk_id=chunk.chunk_id,
            agent=self.name,
            confidence=round(confidence, 3),
            summary=summary,
            metadata=metadata,
        )

    def _describe_views(self, views: Sequence[VisionView]) -> List[str]:
        descriptions: List[str] = []
        for view in views:
            caption = None
            if self._vlm and view.path.exists():  # pragma: no branch - quick guard
                try:
                    caption = self._vlm.generate(view.path)
                except Exception:
                    caption = None
            descriptions.append(caption or f"{view.role}:{view.augmentation}")
        return descriptions

    def _summarize(self, chunk: common.Chunk, descriptions: Sequence[str]) -> str:
        prompt = self._build_prompt(chunk, descriptions)
        if not prompt:
            return "".join(descriptions)
        client = self._llm_client
        if client is None:
            return prompt
        try:  # pragma: no cover - depends on vLLM availability
            summary = client.generate(prompt, max_tokens=192, temperature=0.0).strip()
        except Exception:
            summary = ""
        return summary or prompt

    def _extra_metadata(self, chunk: common.Chunk, _: Sequence[str]) -> Dict[str, object]:
        return {"region_type": chunk.metadata.get("region_type") if chunk.metadata else None}

    @abstractmethod
    def _build_prompt(self, chunk: common.Chunk, descriptions: Sequence[str]) -> str:
        raise NotImplementedError

    @abstractmethod
    def _score(self, chunk: common.Chunk, descriptions: Sequence[str]) -> float:
        raise NotImplementedError


class StructuralVisionAgent(BaseVisionAgent):
    def __init__(self) -> None:
        prompt = (
            "Summarize the document layout visible in the following views. "
            "Focus on columns, headings, indentation, and relative positioning."
        )
        super().__init__("StructuralVisionAgent", prompt)

    def _build_prompt(self, chunk: common.Chunk, descriptions: Sequence[str]) -> str:
        return (
            f"Chunk {chunk.chunk_id} layout cues:\n" + "\n".join(f"- {desc}" for desc in descriptions if desc)
        )

    def _score(self, chunk: common.Chunk, descriptions: Sequence[str]) -> float:
        keywords = ("column", "grid", "heading", "indent", "margin")
        text = " ".join(desc.lower() for desc in descriptions)
        hits = sum(text.count(keyword) for keyword in keywords)
        bonus = 0.1 if (chunk.metadata or {}).get("region_type") in {"section", "table"} else 0.0
        return _clamp(hits / max(1, len(descriptions)) + bonus)


class MathematicalVisionAgent(BaseVisionAgent):
    def __init__(self) -> None:
        prompt = (
            "List the mathematical structures observed in the views. Identify equations, symbols, or graphs."
        )
        super().__init__("MathematicalVisionAgent", prompt)

    def _build_prompt(self, chunk: common.Chunk, descriptions: Sequence[str]) -> str:
        return (
            f"Chunk {chunk.chunk_id} math characteristics:\n" + "\n".join(f"- {desc}" for desc in descriptions if desc)
        )

    def _score(self, chunk: common.Chunk, descriptions: Sequence[str]) -> float:
        math_keywords = ("equation", "formula", "symbol", "integral", "matrix", "graph")
        text = " ".join(desc.lower() for desc in descriptions)
        hits = sum(text.count(keyword) for keyword in math_keywords)
        meta = chunk.metadata or {}
        if meta.get("region_type") in {"equation", "formula"}:
            hits += 1
        if meta.get("math_role"):
            hits += 0.5
        return _clamp(hits / max(1, len(descriptions)))


class SemanticVisionAgent(BaseVisionAgent):
    def __init__(self) -> None:
        prompt = (
            "Describe the semantic intent of the content shown (question, paragraph, definition, code, etc.)."
        )
        super().__init__("SemanticVisionAgent", prompt)

    def _build_prompt(self, chunk: common.Chunk, descriptions: Sequence[str]) -> str:
        context = (chunk.metadata or {}).get("section_title") or "context"
        header = f"Chunk {chunk.chunk_id} semantics ({context}):\n"
        return header + "\n".join(f"- {desc}" for desc in descriptions if desc)

    def _score(self, chunk: common.Chunk, descriptions: Sequence[str]) -> float:
        tokens = ("question", "answer", "definition", "paragraph", "code")
        text = " ".join(desc.lower() for desc in descriptions)
        hits = sum(text.count(token) for token in tokens)
        if (chunk.metadata or {}).get("question_label"):
            hits += 0.5
        return _clamp(hits / max(1, len(descriptions)))


class StyleVisionAgent(BaseVisionAgent):
    def __init__(self) -> None:
        prompt = (
            "Assess typography elements visible in the snippet such as font weight, alignment, and spacing."  # noqa: E501
        )
        super().__init__("StyleVisionAgent", prompt)

    def _build_prompt(self, chunk: common.Chunk, descriptions: Sequence[str]) -> str:
        return (
            f"Chunk {chunk.chunk_id} style cues:\n" + "\n".join(f"- {desc}" for desc in descriptions if desc)
        )

    def _score(self, chunk: common.Chunk, descriptions: Sequence[str]) -> float:
        style_keywords = ("bold", "italic", "spacing", "alignment", "margin")
        text = " ".join(desc.lower() for desc in descriptions)
        hits = sum(text.count(keyword) for keyword in style_keywords)
        notes = chunk.metadata or {}
        if notes.get("font_variant"):
            hits += 0.5
        return _clamp(hits / max(1, len(descriptions)))


class VisionAgentSuite:
    """Convenience wrapper that runs multiple vision agents in sequence."""

    def __init__(self, agents: Iterable[BaseVisionAgent] | None = None) -> None:
        self.agents: List[BaseVisionAgent] = list(agents) if agents else [
            StructuralVisionAgent(),
            MathematicalVisionAgent(),
            SemanticVisionAgent(),
            StyleVisionAgent(),
        ]

    def evaluate(self, chunk: common.Chunk, views: Sequence[VisionView]) -> List[VisionAgentResult]:
        results: List[VisionAgentResult] = []
        for agent in self.agents:
            try:
                results.append(agent.analyze(chunk, views))
            except Exception as exc:  # pragma: no cover - defensive
                results.append(
                    VisionAgentResult(
                        chunk_id=chunk.chunk_id,
                        agent=agent.name,
                        confidence=0.0,
                        summary=f"agent-error: {exc}",
                        metadata={"error": True},
                    )
                )
        return results


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


__all__ = [
    "BaseVisionAgent",
    "SemanticVisionAgent",
    "StructuralVisionAgent",
    "MathematicalVisionAgent",
    "StyleVisionAgent",
    "VisionAgentResult",
    "VisionAgentSuite",
]
