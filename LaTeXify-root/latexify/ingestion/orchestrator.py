from __future__ import annotations

import asyncio
import difflib
import json
import os
import re
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from latexify.utils.logging import log_info, log_warning

from .backends.base import BaseCouncilBackend, CouncilOutput, LayoutChunk

PLACEHOLDER_SNIPPETS = {
    "",
    "(no text extracted)",
    "\\[ \\text{No equations detected} \\]",
}

CHEM_RX = re.compile(r"\b(?:[A-Z][a-z]?\d{0,3}){2,}\b")
MATH_RX = re.compile(r"(\$\$|\\\[|\\\(|\\begin\{equation|\\sum|\\int)")
TABLE_RX = re.compile(r"\\begin\{tabular|\|.*\|")
CODE_RX = re.compile(r"```|\bclass\b|\bdef \b|\bfor \(|\{\s*$", re.MULTILINE)
FIG_RX = re.compile(r"figure|diagram|plot|graph|image", re.IGNORECASE)


def classify_snippet(text: str) -> Dict[str, Any]:
    snippet = (text or "").strip()
    lower = snippet.lower()
    if not snippet:
        return {"content_type": "text", "confidence": 0.2, "explanation": "Empty snippet; defaulting to text."}
    if TABLE_RX.search(snippet):
        return {"content_type": "table", "confidence": 0.9, "explanation": "Detected tabular delimiters."}
    if MATH_RX.search(snippet):
        return {"content_type": "math", "confidence": 0.9, "explanation": "LaTeX math markers present."}
    if CODE_RX.search(snippet):
        return {"content_type": "code", "confidence": 0.75, "explanation": "Code fences/keywords detected."}
    if CHEM_RX.search(snippet):
        return {"content_type": "chem", "confidence": 0.7, "explanation": "Chemical formula pattern detected."}
    if FIG_RX.search(lower):
        return {"content_type": "figure_with_text", "confidence": 0.65, "explanation": "Figure keywords detected."}
    return {"content_type": "text", "confidence": 0.5, "explanation": "Defaulted to prose."}


def _is_meaningful_text(text: str) -> bool:
    stripped = (text or "").strip()
    if stripped in PLACEHOLDER_SNIPPETS:
        return False
    if len(stripped) < 4:
        return False
    letters = sum(ch.isalpha() for ch in stripped)
    return letters >= 2


def _now_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class CouncilOrchestrator:
    def __init__(
        self,
        backends: Sequence[BaseCouncilBackend],
        run_dir: Path,
        *,
        chunk_assets: Mapping[str, Sequence[Path]] | None = None,
    ) -> None:
        self.backends = list(backends)
        self.run_dir = run_dir
        self.records: List[Dict[str, Any]] = []
        self.records_by_chunk: Dict[str, List[Dict[str, Any]]] = {}
        self.permissive_backends = [backend.name for backend in self.backends if getattr(backend, "permissive", False)]
        self.stats: Dict[str, int] = {
            "backend_warnings": 0,
            "fallback_hits": 0,
            "permissive_hits": 0,
            "consensus_recovered": 0,
            "empty_consensus": 0,
        }
        self.recoveries: List[Dict[str, Any]] = []
        self.chunk_assets: Dict[str, List[str]] = {}
        if chunk_assets:
            for chunk_id, paths in chunk_assets.items():
                stringified = [str(path) for path in paths if path]
                if stringified:
                    self.chunk_assets[str(chunk_id)] = stringified
        self._merge_client: Optional[Callable[[str], str]] = None

    async def process_chunks(self, chunks: Sequence[LayoutChunk]) -> None:
        outputs_root = self.run_dir / "outputs"
        outputs_root.mkdir(parents=True, exist_ok=True)
        for chunk in chunks:
            tasks = [backend.process(chunk) for backend in self.backends]
            results = await asyncio.gather(*tasks)
            for result in results:
                self._persist(result)

    def _persist(self, result: CouncilOutput) -> None:
        backend_dir = self.run_dir / "outputs" / result.backend
        backend_dir.mkdir(parents=True, exist_ok=True)
        metadata = result.metadata or {}
        payload = {
            "backend": result.backend,
            "chunk_id": result.chunk_id,
            "page_index": result.page_index,
            "text": result.text,
            "confidence": result.confidence,
            "metadata": metadata,
        }
        out_path = backend_dir / f"{result.chunk_id}.json"
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        self.records.append(
            {
                "backend": result.backend,
                "chunk_id": result.chunk_id,
                "page_index": result.page_index,
                "path": str(out_path),
                "confidence": result.confidence,
                "metadata": metadata,
            }
        )
        self.records_by_chunk.setdefault(result.chunk_id, []).append(payload)
        if metadata.get("warning"):
            self.stats["backend_warnings"] += 1
        if metadata.get("fallback_used"):
            self.stats["fallback_hits"] += 1
            self.recoveries.append(
                {
                    "chunk_id": result.chunk_id,
                    "backend": result.backend,
                    "reason": metadata.get("reason") or metadata.get("warning") or "fallback",
                }
            )
        if metadata.get("source") == "generic_ocr":
            self.stats["permissive_hits"] += 1

    def write_manifest(self) -> None:
        council_dir = self.run_dir / "council"
        council_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "created_at": _now_ts(),
            "records": self.records,
            "backends": [backend.name for backend in self.backends],
        }
        (council_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def write_consensus(self) -> Path:
        consensus_dir = self.run_dir / "consensus"
        consensus_dir.mkdir(parents=True, exist_ok=True)
        chunk_ids: List[str] = []
        for chunk_id, outputs in sorted(self.records_by_chunk.items()):
            chunk_ids.append(chunk_id)
            merged = self._merge_chunk(chunk_id, outputs)
            (consensus_dir / f"{chunk_id}.json").write_text(
                json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            (consensus_dir / f"{chunk_id}.txt").write_text(merged["text"], encoding="utf-8")
        (consensus_dir / "manifest.json").write_text(
            json.dumps({"chunks": chunk_ids, "count": len(chunk_ids)}, indent=2),
            encoding="utf-8",
        )
        return consensus_dir

    def write_resilience_report(self) -> Path:
        report = {
            "created_at": _now_ts(),
            "chunks": len(self.records_by_chunk),
            "permissive_backends": self.permissive_backends,
            "stats": {k: int(v) for k, v in self.stats.items()},
            "recoveries": self.recoveries,
        }
        path = self.run_dir / "resilience_report.json"
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return path

    def _merge_chunk(self, chunk_id: str, outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not outputs:
            self.stats["empty_consensus"] += 1
            return {
                "chunk_id": chunk_id,
                "text": "",
                "content_type": "text",
                "classification_confidence": 0.0,
                "classification_explanation": "No backend output available",
                "source_backend": "",
                "supporting_backends": [],
                "ocr_outputs": {},
                "assets": self.chunk_assets.get(chunk_id, []),
            }
        support_scores = self._support_scores(outputs)
        best_idx = self._select_best(outputs, support_scores)
        selected = outputs[best_idx]
        merged_text = selected["text"]
        disagreement = support_scores[best_idx] < 0.55 if support_scores else False
        if not _is_meaningful_text(merged_text):
            fallback = next((entry for entry in outputs if _is_meaningful_text(entry["text"])), None)
            if fallback:
                selected = fallback
                merged_text = fallback["text"]
                self.stats["consensus_recovered"] += 1
                self.recoveries.append(
                    {
                        "chunk_id": chunk_id,
                        "backend": fallback["backend"],
                        "reason": "consensus_replacement",
                    }
                )
                disagreement = False
            else:
                self.stats["empty_consensus"] += 1
        elif disagreement:
            llm_merged = self._llm_merge(chunk_id, outputs)
            if llm_merged:
                merged_text = llm_merged
                self.stats["consensus_recovered"] += 1
            else:
                fallback = next((entry for entry in outputs if _is_meaningful_text(entry["text"])), None)
                if fallback and fallback is not selected:
                    selected = fallback
                    merged_text = fallback["text"]
                    self.stats["consensus_recovered"] += 1
                    self.recoveries.append(
                        {
                            "chunk_id": chunk_id,
                            "backend": fallback["backend"],
                            "reason": "consensus_replacement",
                        }
                    )
                else:
                    self.stats["empty_consensus"] += 1
        ocr_outputs = {entry["backend"]: entry["text"] for entry in outputs}
        classification = classify_snippet(selected["text"])
        asset_paths = self.chunk_assets.get(chunk_id, [])
        return {
            "chunk_id": chunk_id,
            "text": merged_text,
            "content_type": classification["content_type"],
            "classification_confidence": classification["confidence"],
            "classification_explanation": classification["explanation"],
            "source_backend": selected["backend"],
            "supporting_backends": [entry["backend"] for entry in outputs if entry is not selected],
            "ocr_outputs": ocr_outputs,
            "assets": asset_paths,
        }

    def _support_scores(self, outputs: Sequence[Dict[str, Any]]) -> List[float]:
        scores: List[float] = []
        for idx, entry in enumerate(outputs):
            sims: List[float] = []
            for jdx, other in enumerate(outputs):
                if idx == jdx:
                    continue
                sims.append(self._text_similarity(entry["text"], other["text"]))
            scores.append(statistics.fmean(sims) if sims else 0.0)
        return scores

    @staticmethod
    def _text_similarity(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        return difflib.SequenceMatcher(None, a, b).ratio()

    def _select_best(self, outputs: List[Dict[str, Any]], support_scores: Sequence[float]) -> int:
        def score(entry: Dict[str, Any], support: float) -> float:
            text = entry["text"]
            confidence = float(entry.get("confidence") or 0.0)
            text_bonus = 0.0
            if "\\begin{tabular" in text or "|" in text:
                text_bonus += 0.05
            if "\\[" in text or "$$" in text:
                text_bonus += 0.05
            if not text.strip():
                text_bonus -= 0.5
            return 0.65 * support + 0.35 * confidence + text_bonus

        best_idx = 0
        best_score = float("-inf")
        for idx, entry in enumerate(outputs):
            s = score(entry, support_scores[idx] if support_scores else 0.0)
            if s > best_score:
                best_score = s
                best_idx = idx
        return best_idx

    def _llm_merge(self, chunk_id: str, outputs: Sequence[Dict[str, Any]]) -> Optional[str]:
        client = self._resolve_merge_client()
        if client is None:
            return None
        prompt_lines = [
            "You are the consensus arbiter for OCR outputs.",
            "Return a single best-effort paragraph that merges the faithful text.",
            "Chunk examples:",
        ]
        for entry in outputs:
            prompt_lines.append(f"- Backend={entry['backend']}: {entry['text'][:1200]}")
        prompt_lines.append("Merged snippet:")
        prompt = "\n".join(prompt_lines)
        try:
            response = client(prompt)
        except Exception as exc:  # pragma: no cover - runtime dependency
            log_warning("LLM merge failed", chunk_id=chunk_id, error=str(exc))
            return None
        merged = (response or "").strip()
        return merged or None

    def _resolve_merge_client(self) -> Optional[Callable[[str], str]]:
        if self._merge_client is not None:
            return self._merge_client
        model_path = os.environ.get("LATEXIFY_COUNCIL_MERGE_MODEL")
        if not model_path:
            self._merge_client = None
            return None
        try:
            from latexify.pipeline.model_backends import LlamaCppBackend, LlamaCppConfig
        except Exception as exc:  # pragma: no cover - llama backend optional
            log_warning("Merge model backend unavailable", error=str(exc))
            self._merge_client = None
            return None
        cfg = LlamaCppConfig(
            model_path=Path(model_path).expanduser(),
            n_ctx=2048,
            n_batch=256,
            seed=1337,
            tensor_split="auto",
            verbose=False,
        )
        backend = LlamaCppBackend(cfg)

        def _client(prompt: str) -> str:
            return backend.generate(prompt, max_tokens=256, temperature=0.05, top_p=0.8, top_k=40, stop=["Merged snippet:"])

        self._merge_client = _client
        return self._merge_client


class IngestionPipeline:
    """Encapsulates the ingestion workflow end-to-end."""

    def __init__(
        self,
        *,
        pdf: Path,
        run_dir: Path,
        chunks: Sequence[LayoutChunk],
        backends: Sequence[BaseCouncilBackend],
    ) -> None:
        self.pdf = pdf
        self.run_dir = run_dir
        self.chunks = list(chunks)
        self.backends = list(backends)
        self._chunk_assets: Dict[str, List[Path]] = {}
        for chunk in self.chunks:
            if chunk.image_path:
                self._chunk_assets.setdefault(chunk.chunk_id, []).append(chunk.image_path)
        self._council = CouncilOrchestrator(self.backends, run_dir, chunk_assets=self._chunk_assets)

    async def run_async(self) -> Dict[str, Any]:
        log_info("Starting council orchestrator", run_dir=str(self.run_dir), chunk_count=len(self.chunks))
        await self._council.process_chunks(self.chunks)
        self._council.write_manifest()
        consensus_dir = self._council.write_consensus()
        resilience_report = self._council.write_resilience_report()
        return {
            "source_pdf": str(self.pdf),
            "chunk_count": len(self.chunks),
            "consensus_dir": str(consensus_dir),
            "resilience_report": str(resilience_report),
            "backends": [backend.name for backend in self.backends],
        }

    def run(self) -> Dict[str, Any]:
        return asyncio.run(self.run_async())


__all__ = ["CouncilOrchestrator", "IngestionPipeline", "classify_snippet"]
