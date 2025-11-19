"""Unified math support utilities (classifier + environment detector)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

try:  # pragma: no cover - optional dependency
    from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore
except Exception:  # pragma: no cover
    AutoModelForSequenceClassification = None  # type: ignore
    AutoTokenizer = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import torch
except Exception:  # pragma: no cover
    torch = None

INLINE_RE = re.compile(r"\$(?:[^$]|\\\$)+\$")
DISPLAY_RE = re.compile(r"\\begin\{(equation|align|gather|cases)\}|\\\\[|\\\\]")
DERIVATION_RE = re.compile(r"(=|\\equiv|\\approx).*(\\\\|\n)")
PROOF_RE = re.compile(r"(\\Rightarrow|\\therefore|\\because|\\implies)")
FUNC_DEF_RE = re.compile(r"[a-zA-Z]\\?\([^)]+\)\s*=\s*")

INLINE_MATH_RE = re.compile(r"\$(?!\$).+?\$")
DISPLAY_ENV_RE = re.compile(r"\\begin\{(?P<env>[a-z*]+)\}")
ALIGN_HINT_RE = re.compile(r"(=|\\approx|\\sim).*(\\\\|&)")
CASES_HINT_RE = re.compile(r"(cases|piecewise)", re.IGNORECASE)
MATRIX_HINT_RE = re.compile(r"\\begin\{[pbv]?matrix\}")


@dataclass
class ClassificationResult:
    label: str
    score: float


class MathContentClassifier:
    """Rule-first classifier with optional transformer refinement."""

    def __init__(self, enable_model: bool = False, model_name: str = "") -> None:
        self.enable_model = enable_model and AutoModelForSequenceClassification is not None
        self.model_name = model_name or "patrickvonplaten/bert-base-uncased-math-text"
        self._tokenizer = None
        self._model = None
        if self.enable_model and torch is not None:
            try:  # pragma: no cover - heavy dependency
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                self._model.eval()
            except Exception:
                self.enable_model = False
                self._tokenizer = None
                self._model = None
        else:
            self.enable_model = False

    def classify(self, text: str) -> ClassificationResult:
        label, score = self._rules(text)
        if self.enable_model and self._model and self._tokenizer:
            refined = self._model_refine(text)
            if refined and refined.score > score:
                return refined
        return ClassificationResult(label=label, score=score)

    def _rules(self, text: str) -> tuple[str, float]:
        stripped = text.strip()
        if not stripped:
            return "empty", 0.0
        if DISPLAY_RE.search(stripped):
            return "display-equation", 0.9
        if INLINE_RE.search(stripped) and len(stripped) < 256:
            return "inline-equation", 0.8
        if PROOF_RE.search(stripped) or stripped.lower().startswith("proof"):
            return "proof-step", 0.75
        if DERIVATION_RE.search(stripped):
            return "derivation", 0.7
        if FUNC_DEF_RE.search(stripped):
            return "function-definition", 0.65
        if stripped.lower().startswith(("question", "problem")):
            return "question-stem", 0.6
        if stripped.lower().startswith(("solution", "answer")):
            return "solution-text", 0.6
        return "paragraph", 0.4

    def _model_refine(self, text: str) -> Optional[ClassificationResult]:  # pragma: no cover - heavy dep
        if not self._tokenizer or not self._model or torch is None:
            return None
        try:
            tokens = self._tokenizer(text, truncation=True, padding=True, return_tensors="pt")
            with torch.no_grad():
                outputs = self._model(**tokens)
            probabilities = outputs.logits.softmax(dim=-1)[0]
            score, idx = float(probabilities.max().item()), int(probabilities.argmax().item())
            label = self._model.config.id2label.get(idx, "paragraph")
            return ClassificationResult(label=label, score=score)
        except Exception:
            return None


@dataclass
class MathEnvironmentDetector:
    """Heuristically pick the most appropriate LaTeX math environment."""

    prefer_align: bool = True

    def detect(self, block_type: str, text: str, metadata: Dict[str, object] | None = None) -> str | None:
        stripped = text.strip()
        if not stripped:
            return None
        if stripped.startswith("\\[") and stripped.endswith("\\]"):
            return "displaymath"
        existing = DISPLAY_ENV_RE.search(stripped)
        if existing:
            return existing.group("env")
        region = (metadata or {}).get("region_type")
        if region == "table":
            return None
        if MATRIX_HINT_RE.search(stripped):
            return "bmatrix"
        if CASES_HINT_RE.search(stripped):
            return "cases"
        if "&" in stripped or ALIGN_HINT_RE.search(stripped):
            return "align*" if self.prefer_align else "align"
        if "\\" in stripped:
            return "align*"
        if block_type in {"equation", "formula"} or INLINE_MATH_RE.search(stripped):
            return "equation"
        if stripped.startswith("\\left[") and "\\right]" in stripped:
            return "bmatrix"
        return None

    def wrap(self, block_type: str, text: str, metadata: Dict[str, object] | None = None) -> str:
        stripped = text.strip()
        if stripped.startswith("\\[") and stripped.endswith("\\]"):
            return text
        if stripped.startswith("$$") and stripped.endswith("$$"):
            return text
        env = self.detect(block_type, text, metadata)
        if not env:
            return text
        if stripped.startswith("\\begin"):
            return text
        body = stripped
        if env.endswith("*") and not env.startswith("align"):
            env = env.rstrip("*")
        return f"\\begin{{{env}}}\n{body}\n\\end{{{env}}}"


__all__ = [
    "MathContentClassifier",
    "ClassificationResult",
    "MathEnvironmentDetector",
]
