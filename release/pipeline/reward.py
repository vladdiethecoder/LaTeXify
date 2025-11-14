"""Tri-modal reward computation for LaTeX generations."""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

from ..core import common

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - optional heavy dependency
    from bert_score import score as bert_score_fn
except Exception:  # pragma: no cover
    bert_score_fn = None

COMMAND_RE = re.compile(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^}]*\})?")
ENV_RE = re.compile(r"\\(begin|end)\{[^\}]+\}")
MATH_RE = re.compile(r"\$[^$]+\$|\\\[.*?\\\]|\\\(.+?\\\)", re.DOTALL)
CURLY_RE = re.compile(r"[{}]")
WHITESPACE_RE = re.compile(r"\s+")

DEFAULT_WEIGHTS = {"syntax": 0.5, "semantic": 0.3, "aesthetic": 0.2}


def latex_to_text(latex: str) -> str:
    text = re.sub(r"%.*", " ", latex)
    text = re.sub(MATH_RE, " ", text)
    text = re.sub(ENV_RE, " ", text)
    text = re.sub(COMMAND_RE, " ", text)
    text = re.sub(CURLY_RE, " ", text)
    text = re.sub(WHITESPACE_RE, " ", text)
    return text.strip()


def _token_overlap(a: str, b: str) -> float:
    toks_a = set(a.lower().split())
    toks_b = set(b.lower().split())
    if not toks_a or not toks_b:
        return 0.0
    return len(toks_a & toks_b) / len(toks_a | toks_b)


def semantic_score(source_text: str, generated_text: str) -> float:
    if not source_text.strip() or not generated_text.strip():
        return 0.0
    if bert_score_fn is not None:
        try:  # pragma: no cover - heavy dependency
            _, _, f1 = bert_score_fn(
                [generated_text],
                [source_text],
                lang="en",
                rescale_with_baseline=True,
            )
            return float(f1[0])
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("BERTScore failed (%s); falling back to token overlap", exc)
    return _token_overlap(source_text, generated_text)


def syntax_score(validation_data: Dict[str, object]) -> float:
    if validation_data.get("success"):
        return 1.0
    errors = validation_data.get("errors", [])
    if errors:
        # penalize harder when there are explicit LaTeX errors
        return -1.0
    return -0.2


def aesthetic_score(latex: str) -> float:
    score = 0.3
    if "\\usepackage{booktabs}" in latex or "\\toprule" in latex:
        score += 0.2
    if "\\begin{align" in latex or "\\begin{equation}" in latex:
        score += 0.1
    if "\\usepackage{geometry}" in latex and "margin" in latex:
        score += 0.05
    if "\\begin{figure" in latex and "\\includegraphics" in latex:
        score += 0.15
    if "\\usepackage{hyperref}" in latex:
        score += 0.05
    if "\\begin{table}" in latex and "\\caption" in latex:
        score += 0.1
    return max(0.0, min(1.0, score))


def evaluate_rewards(
    chunks_path: Path,
    tex_path: Path,
    validation_path: Path,
    output_path: Path,
    weights: Dict[str, float] | None = None,
    *,
    mode: str = "heuristic",
    trace_path: Path | None = None,
) -> Path:
    if mode not in {"heuristic", "mm"}:
        raise ValueError(f"Unsupported reward mode: {mode}")
    weights = weights or DEFAULT_WEIGHTS
    chunks = common.load_chunks(chunks_path) if chunks_path.exists() else []
    source_text = "\n\n".join(chunk.text for chunk in chunks)
    latex = tex_path.read_text(encoding="utf-8") if tex_path.exists() else ""
    generated_text = latex_to_text(latex)
    validation_data = json.loads(validation_path.read_text(encoding="utf-8")) if validation_path.exists() else {}
    syntax = syntax_score(validation_data)
    semantic = semantic_score(source_text, generated_text)
    if mode == "mm":
        from .reward_mm import aesthetic_mm_score

        try:
            aesthetic = aesthetic_mm_score(tex_path)
        except Exception as exc:  # pragma: no cover - runtime safeguard
            LOGGER.warning("Multimodal reward failed (%s); falling back to heuristic.", exc)
            aesthetic = aesthetic_score(latex)
    else:
        aesthetic = aesthetic_score(latex)
    total = (
        weights.get("syntax", 0.5) * syntax
        + weights.get("semantic", 0.3) * semantic
        + weights.get("aesthetic", 0.2) * aesthetic
    )
    payload = {
        "weights": weights,
        "components": {
            "syntax": syntax,
            "semantic": semantic,
            "aesthetic": aesthetic,
        },
        "reward": total,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if trace_path:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        record = {
            "timestamp": timestamp,
            "components": payload["components"],
            "reward": total,
            "mode": mode,
        }
        with trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
    return output_path


__all__ = ["evaluate_rewards", "latex_to_text"]
