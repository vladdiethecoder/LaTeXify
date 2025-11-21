"""RAG utilities for retrieving high-quality LaTeX environments as exemplars."""
from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

LOGGER = logging.getLogger(__name__)

ENV_PATTERNS = {
    "table": re.compile(r"\\begin\{table\}[\s\S]*?\\end\{table\}", re.MULTILINE),
    "figure": re.compile(r"\\begin\{figure\}[\s\S]*?\\end\{figure\}", re.MULTILINE),
    "equation": re.compile(r"\\begin\{(equation\*?|align\*?|multline)\}[\s\S]*?\\end\{\1\}", re.MULTILINE),
}
PACKAGE_HINTS = {
    "booktabs": ["\\toprule", "\\midrule", "\\bottomrule"],
    "amsmath": ["\\align", "\\begin{align", "\\begin{multline", "\\frac", "\\sum"],
    "graphicx": ["\\includegraphics"],
    "caption": ["\\caption"],
}
EMBED_DIM = 256


def _embed(text: str) -> List[float]:
    vec = [0.0] * EMBED_DIM
    for token in re.findall(r"[A-Za-z]+", text.lower()):
        idx = abs(hash(token)) % EMBED_DIM
        vec[idx] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [round(v / norm, 6) for v in vec]


def _cosine(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(vec_a, vec_b))


def _infer_packages(snippet: str) -> List[str]:
    packages: List[str] = []
    for pkg, markers in PACKAGE_HINTS.items():
        if any(marker in snippet for marker in markers):
            packages.append(pkg)
    return packages


@dataclass
class RAGEntry:
    entry_id: str
    doc_id: str
    snippet_type: str
    text: str
    packages: List[str]
    embedding: List[float]
    domain: str | None = None
    quantization_levels: int | None = None

    def to_json(self) -> Dict[str, object]:
        return {
            "entry_id": self.entry_id,
            "doc_id": self.doc_id,
            "type": self.snippet_type,
            "text": self.text,
            "packages": self.packages,
            "embedding": self.embedding,
            "domain": self.domain,
            "quantization_levels": self.quantization_levels,
        }

    @classmethod
    def from_json(cls, payload: Dict[str, object]) -> "RAGEntry":
        return cls(
            entry_id=payload["entry_id"],
            doc_id=payload["doc_id"],
            snippet_type=payload["type"],
            text=payload["text"],
            packages=payload.get("packages", []),
            embedding=payload.get("embedding", _embed(payload["text"])),
            domain=payload.get("domain"),
            quantization_levels=payload.get("quantization_levels"),
        )

    def embedding_vector(self) -> List[float]:
        if not self.quantization_levels:
            return list(self.embedding)
        scale = max(1, int(self.quantization_levels) - 1)
        return [min(1.0, max(0.0, value / scale)) for value in self.embedding]

    def quantized_copy(self, levels: int) -> "RAGEntry":
        if levels <= 1:
            return self
        if self.quantization_levels == levels:
            return self
        scale = max(1, levels - 1)
        source = self.embedding_vector()
        quantized = [min(scale, max(0, int(round(value * scale)))) for value in source]
        return replace(self, embedding=quantized, quantization_levels=levels)

    def footprint_bytes(self) -> int:
        text_bytes = len(self.text.encode("utf-8"))
        per_value = 1 if self.quantization_levels else 8
        embed_bytes = len(self.embedding) * per_value
        aux = 64 + len(self.packages) * 8
        return text_bytes + embed_bytes + aux


class RAGIndex:
    def __init__(self, entries: Iterable[RAGEntry]) -> None:
        self._entries: List[RAGEntry] = list(entries)
        self._by_type: Dict[str, List[RAGEntry]] = {}
        for entry in self._entries:
            self._by_type.setdefault(entry.snippet_type, []).append(entry)

    def search(
        self,
        query: str,
        snippet_type: str | None,
        k: int = 3,
        domain: str | None = None,
    ) -> List[RAGEntry]:
        if not query.strip() or not self._entries:
            return []
        vec = _embed(query)
        candidates = self._by_type.get(snippet_type, self._entries) if snippet_type else self._entries
        scored = []
        for entry in candidates:
            score = _cosine(vec, entry.embedding_vector())
            if domain:
                if entry.domain == domain:
                    score += 0.05
                elif entry.domain:
                    score -= 0.02
            scored.append((score, entry))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [entry for _, entry in scored[:k] if _ > 0.0]

    @classmethod
    def load(cls, path: Path) -> "RAGIndex":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(RAGEntry.from_json(item) for item in data)

    def to_json(self) -> List[Dict[str, object]]:
        return [entry.to_json() for entry in self._entries]

    def entries(self) -> List[RAGEntry]:
        return list(self._entries)


def _infer_domain(tex_path: Path, source_dir: Path) -> str | None:
    try:
        relative = tex_path.relative_to(source_dir)
    except ValueError:
        return None
    parts = relative.parts
    if len(parts) > 1:
        return parts[0]
    return None


def extract_environments(tex_path: Path, source_dir: Path | None = None) -> List[RAGEntry]:
    if not tex_path.exists():
        return []
    text = tex_path.read_text(encoding="utf-8", errors="ignore")
    domain = _infer_domain(tex_path, source_dir) if source_dir else None
    entries: List[RAGEntry] = []
    for snippet_type, pattern in ENV_PATTERNS.items():
        for idx, match in enumerate(pattern.finditer(text)):
            snippet = match.group(0).strip()
            entry = RAGEntry(
                entry_id=f"{tex_path.stem}-{snippet_type}-{idx:03d}",
                doc_id=tex_path.name,
                snippet_type=snippet_type,
                text=snippet,
                packages=_infer_packages(snippet),
                embedding=_embed(snippet),
                domain=domain,
            )
            entries.append(entry)
    return entries


def build_index(source_dir: Path, output_path: Path) -> Path:
    entries: List[RAGEntry] = []
    if not source_dir.exists():
        LOGGER.info("RAG source directory %s missing; skipping index build", source_dir)
        return output_path
    for tex_file in source_dir.rglob("*.tex"):
        entries.extend(extract_environments(tex_file, source_dir))
    if not entries:
        LOGGER.warning("No LaTeX environments found under %s", source_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps([entry.to_json() for entry in entries], indent=2), encoding="utf-8")
    LOGGER.info("RAG index built with %s entries", len(entries))
    return output_path


def _deduplicate_entries(entries: Iterable[RAGEntry]) -> List[RAGEntry]:
    seen: set[tuple[str, str]] = set()
    unique: List[RAGEntry] = []
    for entry in entries:
        signature = (entry.snippet_type, entry.text.strip())
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(entry)
    return unique


def optimize_entries(
    entries: Iterable[RAGEntry],
    *,
    budget_bytes: int | None = None,
    quantization_levels: int | None = 64,
) -> List[RAGEntry]:
    prioritized: List[RAGEntry] = []
    optional: List[RAGEntry] = []
    for entry in _deduplicate_entries(entries):
        if entry.snippet_type in {"equation", "table", "figure"} or entry.domain:
            prioritized.append(entry)
        else:
            optional.append(entry)
    ordered = prioritized + optional
    total = 0
    result: List[RAGEntry] = []
    for entry in ordered:
        optimized = entry.quantized_copy(quantization_levels or 0)
        size = optimized.footprint_bytes()
        if budget_bytes is not None and total and total + size > budget_bytes:
            continue
        result.append(optimized)
        total += size
        if budget_bytes is not None and total >= budget_bytes:
            break
    return result


def optimize_index(
    index: RAGIndex,
    *,
    budget_mb: int | None = None,
    quantization_levels: int | None = 64,
) -> RAGIndex:
    budget_bytes = None
    if budget_mb:
        budget_bytes = max(1, budget_mb) * 1024**2
    optimized_entries = optimize_entries(index.entries(), budget_bytes=budget_bytes, quantization_levels=quantization_levels)
    if len(optimized_entries) == len(index.entries()):
        return index
    LOGGER.info(
        "RAG cache optimized: %s → %s entries (budget=%s MB).",
        len(index.entries()),
        len(optimized_entries),
        budget_mb or "unbounded",
    )
    return RAGIndex(optimized_entries)


def load_or_build_index(source_dir: Path, cache_path: Path | None = None) -> RAGIndex:
    if cache_path and cache_path.exists():
        return RAGIndex.load(cache_path)
    if cache_path:
        build_index(source_dir, cache_path)
        if cache_path.exists():
            return RAGIndex.load(cache_path)
    entries = []
    if source_dir.exists():
        for tex_file in source_dir.rglob("*.tex"):
            entries.extend(extract_environments(tex_file, source_dir))
    return RAGIndex(entries)


__all__ = [
    "build_index",
    "extract_environments",
    "load_or_build_index",
    "RAGIndex",
    "RAGEntry",
    "optimize_entries",
    "optimize_index",
]
