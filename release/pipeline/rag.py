"""RAG utilities for retrieving high-quality LaTeX environments as exemplars."""
from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
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

    def to_json(self) -> Dict[str, object]:
        return {
            "entry_id": self.entry_id,
            "doc_id": self.doc_id,
            "type": self.snippet_type,
            "text": self.text,
            "packages": self.packages,
            "embedding": self.embedding,
            "domain": self.domain,
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
        )


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
            score = _cosine(vec, entry.embedding)
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


__all__ = ["build_index", "extract_environments", "load_or_build_index", "RAGIndex", "RAGEntry"]
