"""Unified reward helpers (heuristic + multimodal + consistency)."""
from __future__ import annotations

import json
import logging
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from ..core import common
from ..models.kimi_k2_adapter import get_kimi_adapter
from .cross_validation import run_cross_validation

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
REWARD_PAGE_LIMIT = int(os.environ.get("LATEXIFY_MM_REWARD_PAGE_LIMIT", "4"))
KIMI_REWARD_TEMPERATURE = float(os.environ.get("LATEXIFY_MM_REWARD_TEMPERATURE", "0.05"))
KIMI_REWARD_MAX_TOKENS = int(os.environ.get("LATEXIFY_MM_REWARD_MAX_TOKENS", "192"))
MAX_LATEX_FRAGMENT = int(os.environ.get("LATEXIFY_MM_REWARD_LATEX_CHARS", "4000"))
MAX_SUMMARY_CHARS = int(os.environ.get("LATEXIFY_MM_REWARD_SUMMARY_CHARS", "2000"))

REWARD_PROMPT = (
    "You are a meticulous LaTeX layout judge. Review the provided LaTeX and its plain-text approximation, "
    "then rate the aesthetics, readability, and structure."
    "\nRespond strictly with:\nscore: <decimal between 0 and 1>\nreason: <one sentence justification>."
)

ROOT = Path(__file__).resolve().parents[2]

LAYOUT_FEATURES = {
    "booktabs tables": ("\\usepackage{booktabs}", "\\toprule"),
    "aligned math": ("\\begin{align", "\\begin{equation}"),
    "geometry": ("\\usepackage{geometry}", "margin"),
    "figures": ("\\begin{figure}", "\\includegraphics"),
    "hyperref": ("\\usepackage{hyperref}",),
    "tables": ("\\begin{table}", "\\caption"),
}


def _describe_layout_features(latex: str) -> list[str]:
    hits: list[str] = []
    for label, needles in LAYOUT_FEATURES.items():
        if any(marker in latex for marker in needles):
            hits.append(label)
    return hits


def _build_reward_prompt(latex: str, summary: str, pages: Sequence[int] | None) -> str:
    truncated_latex = latex[:MAX_LATEX_FRAGMENT]
    truncated_summary = summary[:MAX_SUMMARY_CHARS]
    candidate_pages = sorted({page for page in (pages or [1]) if page and page > 0}) or [1]
    highlights = _describe_layout_features(latex)
    features_line = ", ".join(highlights) if highlights else "none"
    return (
        f"{REWARD_PROMPT}\n\n"
        f"Pages prioritized: {candidate_pages}\n"
        f"Layout cues detected: {features_line}\n\n"
        f"LaTeX snippet (truncated):\n{truncated_latex}\n\n"
        f"Plain-text approximation:\n{truncated_summary}\n"
    )


def _extract_score(response: str) -> float:
    match = re.search(r"([-+]?\d*\.\d+|\d+)", response)
    if not match:
        LOGGER.debug("Reward backend returned unparseable response: %s", response)
        return 0.0
    try:
        value = float(match.group(0))
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, value))
def aesthetic_mm_score(tex_path: Path, pages: Sequence[int] | None = None, dpi: int = 150) -> float:
    del dpi  # dpi retained for backwards compatibility with previous signature
    adapter = get_kimi_adapter()
    if adapter is None:
        LOGGER.debug("Kimi adapter unavailable; skipping multimodal reward.")
        return 0.0
    if not tex_path.exists():
        return 0.0
    latex = tex_path.read_text(encoding="utf-8")
    if not latex.strip():
        return 0.0
    summary = latex_to_text(latex)
    prompt = _build_reward_prompt(latex, summary, pages)
    try:
        response = adapter.generate(
            prompt,
            max_tokens=KIMI_REWARD_MAX_TOKENS,
            temperature=KIMI_REWARD_TEMPERATURE,
        )
    except Exception as exc:  # pragma: no cover - depends on llama.cpp runtime
        LOGGER.debug("Kimi reward scoring failed: %s", exc)
        return 0.0
    return _extract_score(response)


def _select_reward_pages(chunks: Sequence[common.Chunk], limit: int) -> List[int]:
    if limit <= 0:
        return []
    pages: List[int] = []
    seen = set()

    def _append(page: int) -> None:
        if page <= 0 or page in seen or len(pages) >= limit:
            return
        pages.append(page)
        seen.add(page)

    _append(1)
    math_counter: Counter[int] = Counter()
    layout_counter: Counter[int] = Counter()
    for chunk in chunks:
        meta = chunk.metadata or {}
        region = meta.get("region_type")
        if meta.get("formula_detected") or region in {"formula", "equation"}:
            math_counter[chunk.page] += 1
        if region in {"table", "figure"}:
            layout_counter[chunk.page] += 1
    for page, _ in math_counter.most_common():
        _append(page)
    for page, _ in layout_counter.most_common():
        _append(page)
    return pages


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
    pages_to_score = _select_reward_pages(chunks, REWARD_PAGE_LIMIT)
    source_text = "\n\n".join(chunk.text for chunk in chunks)
    latex = tex_path.read_text(encoding="utf-8") if tex_path.exists() else ""
    generated_text = latex_to_text(latex)
    validation_data = json.loads(validation_path.read_text(encoding="utf-8")) if validation_path.exists() else {}
    syntax = syntax_score(validation_data)
    semantic = semantic_score(source_text, generated_text)
    if mode == "mm":
        try:
            aesthetic = aesthetic_mm_score(tex_path, pages=pages_to_score)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Multimodal reward failed (%s); falling back to heuristic.", exc)
            aesthetic = aesthetic_score(latex)
    else:
        aesthetic = aesthetic_score(latex)
    pdf_candidate = tex_path.with_suffix(".pdf")
    pdf_path = pdf_candidate if pdf_candidate.exists() else None
    cross_report = run_cross_validation(
        chunks,
        latex,
        validation_data,
        pdf_path=pdf_path,
        source_text=source_text,
    )
    cross_score = cross_report.overall_score
    total = (
        weights.get("syntax", 0.5) * syntax
        + weights.get("semantic", 0.3) * semantic
        + weights.get("aesthetic", 0.2) * aesthetic
    )
    if cross_score:
        total = 0.9 * total + 0.1 * cross_score
    payload = {
        "weights": weights,
        "components": {
            "syntax": syntax,
            "semantic": semantic,
            "aesthetic": aesthetic,
            "cross_validation": cross_score,
        },
        "reward": total,
        "cross_validation": cross_report.to_dict(),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if trace_path:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "components": payload["components"],
            "reward": total,
            "mode": mode,
        }
        with trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
    return output_path


__all__ = [
    "evaluate_rewards",
    "latex_to_text",
    "aesthetic_mm_score",
]
