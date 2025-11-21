"""Comparative evaluation across pipeline branches."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from ..core import common


def evaluate_branches(
    chunks_path: Path,
    snippets_path: Path,
    branch_manifest_path: Path | None,
    latex_image_summary: Dict[str, object] | None,
    vision_summary: Dict[str, object] | None,
    output_path: Path,
) -> Dict[str, object]:
    chunks = {chunk.chunk_id: chunk for chunk in common.load_chunks(chunks_path)} if chunks_path.exists() else {}
    snippets = common.load_snippets(snippets_path) if snippets_path.exists() else []
    bleu_scores: List[float] = []
    low_chunks: List[str] = []
    for snippet in snippets:
        chunk = chunks.get(snippet.chunk_id)
        if not chunk:
            continue
        score = _simple_bleu(chunk.text or "", snippet.latex or "")
        bleu_scores.append(score)
        if score < 0.35:
            low_chunks.append(snippet.chunk_id)
    avg_bleu = round(sum(bleu_scores) / len(bleu_scores), 3) if bleu_scores else 0.0
    manifest = _load_manifest(branch_manifest_path)
    summary = manifest.get("summary") if isinstance(manifest, dict) else {}
    total = max(1, int(summary.get("total", len(manifest.get("results", [])) if isinstance(summary, dict) else 0)))
    success_rate = 0.0
    if isinstance(summary, dict):
        success_rate = float(summary.get("completed", 0)) / total
    visual_metrics = latex_image_summary.get("metrics", {}) if isinstance(latex_image_summary, dict) else {}
    vision_cov = (vision_summary or {}).get("chunk_coverage") if isinstance(vision_summary, dict) else None
    metrics = {
        "avg_bleu": avg_bleu,
        "compile_success_rate": round(success_rate, 3),
        "visual_fidelity": round(
            visual_metrics.get("rendered", 0) / max(1, visual_metrics.get("requested", 0)), 3
        ) if visual_metrics else 0.0,
    }
    if vision_cov is not None:
        metrics["vision_coverage"] = vision_cov
    report = {
        "metrics": metrics,
        "low_bleu_chunks": low_chunks,
        "manifest": summary,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _load_manifest(path: Path | None) -> Dict[str, object]:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _simple_bleu(reference: str, candidate: str, ngram: int = 2) -> float:
    ref_tokens = _tokenize(reference)
    cand_tokens = _tokenize(candidate)
    if not ref_tokens or not cand_tokens:
        return 0.0
    precisions: List[float] = []
    for n in range(1, ngram + 1):
        ref_counts = _ngram_counts(ref_tokens, n)
        cand_counts = _ngram_counts(cand_tokens, n)
        overlap = 0
        total = sum(cand_counts.values()) or 1
        for gram, count in cand_counts.items():
            overlap += min(count, ref_counts.get(gram, 0))
        precisions.append(overlap / total)
    precision = sum(precisions) / len(precisions)
    brevity = min(1.0, len(cand_tokens) / max(1, len(ref_tokens)))
    return round(precision * brevity, 3)


def _tokenize(text: str) -> List[str]:
    return [token.lower() for token in text.replace("\\n", " ").split() if token.strip()]


def _ngram_counts(tokens: Iterable[str], n: int) -> Dict[Tuple[str, ...], int]:
    counts: Dict[Tuple[str, ...], int] = {}
    window = list(tokens)
    for idx in range(len(window) - n + 1):
        gram = tuple(window[idx : idx + n])
        counts[gram] = counts.get(gram, 0) + 1
    return counts


__all__ = ["evaluate_branches"]
