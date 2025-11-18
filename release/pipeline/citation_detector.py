"""Citation pattern detection utilities."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence

from ..core import common


NUMERIC_RE = re.compile(r"\[(\d+(?:\s*(?:-|,)\s*\d+)*)\]")
AUTHOR_YEAR_RE = re.compile(r"\((?P<author>[A-Z][A-Za-z\-\s']+?),\s*(?P<year>19\d{2}|20\d{2})\)")
LATEX_CITE_RE = re.compile(r"\\cite[t|p|alp|author|year]*\{([^}]+)\}")


@dataclass
class CitationMention:
    chunk_id: str
    raw_text: str
    surrounding_text: str


@dataclass
class CitationEntry:
    identifier: str
    raw_text: str
    style: str
    mentions: List[CitationMention] = field(default_factory=list)
    pattern: str | None = None
    latex_key: str | None = None

    def to_json(self) -> Dict[str, object]:
        return {
            "id": self.identifier,
            "raw": self.raw_text,
            "style": self.style,
            "mentions": [mention.__dict__ for mention in self.mentions],
            "pattern": self.pattern,
            "latex_key": self.latex_key,
        }


@dataclass
class CitationReport:
    entries: Dict[str, CitationEntry]
    dominant_style: str

    def to_json(self) -> Dict[str, object]:
        return {
            "dominant_style": self.dominant_style,
            "citations": [entry.to_json() for entry in self.entries.values()],
        }


class CitationDetector:
    """Detect citations in semantic chunks and classify their style."""

    def detect(self, plan: Sequence[common.PlanBlock], chunk_map: Dict[str, common.Chunk]) -> CitationReport:
        entries: Dict[str, CitationEntry] = {}
        style_counts = {"numeric": 0, "author-year": 0, "tex-command": 0}
        for block in plan:
            chunk = chunk_map.get(block.chunk_id)
            if not chunk:
                continue
            text = chunk.text
            entries.update(self._scan_numeric(chunk.chunk_id, text, entries, style_counts))
            entries.update(self._scan_author_year(chunk.chunk_id, text, entries, style_counts))
            entries.update(self._scan_latex(chunk.chunk_id, text, entries, style_counts))
        dominant = max(style_counts, key=style_counts.get) if style_counts else "numeric"
        return CitationReport(entries=entries, dominant_style=dominant)

    def _scan_numeric(
        self,
        chunk_id: str,
        text: str,
        entries: Dict[str, CitationEntry],
        style_counts: Dict[str, int],
    ) -> Dict[str, CitationEntry]:
        for match in NUMERIC_RE.finditer(text):
            raw = match.group(0)
            identifier = f"numeric-{match.group(1).replace(' ', '').replace(',', '_')}"
            entry = entries.get(identifier)
            if not entry:
                entry = CitationEntry(
                    identifier=identifier,
                    raw_text=raw,
                    style="numeric",
                    pattern=re.escape(raw),
                )
                entries[identifier] = entry
            entry.mentions.append(
                CitationMention(
                    chunk_id=chunk_id,
                    raw_text=raw,
                    surrounding_text=self._context_excerpt(text, match.start(), match.end()),
                )
            )
            style_counts["numeric"] += 1
        return entries

    def _scan_author_year(
        self,
        chunk_id: str,
        text: str,
        entries: Dict[str, CitationEntry],
        style_counts: Dict[str, int],
    ) -> Dict[str, CitationEntry]:
        for match in AUTHOR_YEAR_RE.finditer(text):
            author = match.group("author").strip()
            year = match.group("year")
            raw = match.group(0)
            slug = re.sub(r"[^0-9a-z]+", "-", author.lower()).strip("-") or "author"
            identifier = f"authyear-{slug}-{year}"
            entry = entries.get(identifier)
            if not entry:
                entry = CitationEntry(
                    identifier=identifier,
                    raw_text=raw,
                    style="author-year",
                    pattern=re.escape(raw),
                )
                entries[identifier] = entry
            entry.mentions.append(
                CitationMention(
                    chunk_id=chunk_id,
                    raw_text=raw,
                    surrounding_text=self._context_excerpt(text, match.start(), match.end()),
                )
            )
            style_counts["author-year"] += 1
        return entries

    def _scan_latex(
        self,
        chunk_id: str,
        text: str,
        entries: Dict[str, CitationEntry],
        style_counts: Dict[str, int],
    ) -> Dict[str, CitationEntry]:
        for match in LATEX_CITE_RE.finditer(text):
            keys = [key.strip() for key in match.group(1).split(",") if key.strip()]
            for key in keys:
                identifier = f"tex-{key}"
                entry = entries.get(identifier)
                if not entry:
                    entry = CitationEntry(
                        identifier=identifier,
                        raw_text=f"\\cite{{{key}}}",
                        style="tex-command",
                        latex_key=key,
                    )
                    entries[identifier] = entry
                entry.mentions.append(
                    CitationMention(
                        chunk_id=chunk_id,
                        raw_text=match.group(0),
                        surrounding_text=self._context_excerpt(text, match.start(), match.end()),
                    )
                )
                style_counts["tex-command"] += 1
        return entries

    def _context_excerpt(self, text: str, start: int, end: int, window: int = 60) -> str:
        snippet = text[max(0, start - window) : min(len(text), end + window)]
        return " ".join(snippet.split())


__all__ = ["CitationDetector", "CitationReport", "CitationEntry"]
