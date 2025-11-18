"""Bibliography generation from detected citations."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

try:  # pragma: no cover - optional network dependency
    from habanero import Crossref  # type: ignore
except Exception:  # pragma: no cover
    Crossref = None  # type: ignore

from .citation_detector import CitationEntry, CitationReport

LOGGER = logging.getLogger(__name__)


@dataclass
class BibliographyEntry:
    key: str
    title: str
    authors: str
    year: str
    source: str
    raw: str

    def render(self) -> str:
        parts = [self.authors, f"({self.year})" if self.year else ""]
        parts = [part for part in parts if part]
        main = " ".join(parts)
        details = [self.title, self.source]
        details = ", ".join([segment for segment in details if segment])
        return f"{main}. {details}.".strip()


@dataclass
class BibliographyResult:
    latex: str
    packages: List[Dict[str, str | None]]
    preamble_commands: List[str]
    citation_command: str
    replacement_rules: List[Dict[str, str]]
    entries: List[BibliographyEntry]

    def apply_to_text(self, text: str) -> str:
        updated = text
        for rule in self.replacement_rules:
            pattern = re.compile(rule["pattern"])
            replacement = rule["replacement"]

            def _literal_sub(_: re.Match[str], value: str = replacement) -> str:
                return value

            updated = pattern.sub(_literal_sub, updated)
        return updated


class BibliographyGenerator:
    """Create bibliography snippets from detected citations."""

    def __init__(self) -> None:
        self.crossref = Crossref() if Crossref else None

    def build(self, report: CitationReport, style_family: str | None = None) -> BibliographyResult:
        if not report.entries:
            return BibliographyResult(
                latex="",
                packages=[],
                preamble_commands=[],
                citation_command="\\cite",
                replacement_rules=[],
                entries=[],
            )
        style = self._select_style(report, style_family)
        packages, commands, cite_cmd = self._style_packages(style)
        entries: List[BibliographyEntry] = []
        replacement_rules: List[Dict[str, str]] = []
        for idx, entry in enumerate(report.entries.values(), start=1):
            bib_entry = self._build_entry(entry, idx)
            entries.append(bib_entry)
            if entry.style != "tex-command" and entry.pattern:
                replacement_rules.append({
                    "pattern": entry.pattern,
                    "replacement": f"{cite_cmd}{{{bib_entry.key}}}",
                })
        latex_body = self._render_bibliography(entries)
        return BibliographyResult(
            latex=latex_body,
            packages=packages,
            preamble_commands=commands,
            citation_command=cite_cmd,
            replacement_rules=replacement_rules,
            entries=entries,
        )

    def _select_style(self, report: CitationReport, style_family: str | None) -> str:
        if style_family == "ieee":
            return "numeric"
        if style_family in {"acm", "springer"}:
            return "author-year"
        return report.dominant_style or "numeric"

    def _style_packages(self, style: str) -> tuple[List[Dict[str, str | None]], List[str], str]:
        if style == "author-year":
            packages = [{"package": "natbib", "options": "authoryear"}]
            commands = ["\\bibliographystyle{plainnat}"]
            citation_cmd = "\\citep"
        elif style == "tex-command":
            packages = [{"package": "natbib"}]
            commands = ["\\bibliographystyle{plainnat}"]
            citation_cmd = "\\cite"
        else:  # numeric default
            packages = [{"package": "cite"}]
            commands = ["\\bibliographystyle{ieeetr}"]
            citation_cmd = "\\cite"
        return packages, commands, citation_cmd

    def _build_entry(self, citation: CitationEntry, index: int) -> BibliographyEntry:
        key = citation.latex_key or f"ref{index:03d}"
        metadata = self._lookup_metadata(citation.raw_text)
        return BibliographyEntry(
            key=key,
            title=metadata.get("title") or citation.raw_text.strip("()[]"),
            authors=metadata.get("authors") or "Unknown",
            year=metadata.get("year") or "n.d.",
            source=metadata.get("source") or "",
            raw=citation.raw_text,
        )

    def _lookup_metadata(self, query: str) -> Dict[str, str]:
        if not self.crossref:
            return {}
        try:  # pragma: no cover - network dependency
            result = self.crossref.works(query=query, limit=1)
        except Exception as exc:
            LOGGER.debug("Crossref lookup failed for %s: %s", query, exc)
            return {}
        items = result.get("message", {}).get("items", []) if isinstance(result, dict) else []
        if not items:
            return {}
        work = items[0]
        title = " ".join(work.get("title", [])).strip()
        year = None
        if work.get("issued", {}).get("date-parts"):
            year = str(work["issued"]["date-parts"][0][0])
        authors = work.get("author") or []
        author_names = [
            "{} {}".format(author.get("given", "").strip(), author.get("family", "").strip()).strip()
            for author in authors
        ]
        author_text = ", ".join(name for name in author_names if name)
        source_parts = [work.get("container-title", [""])[0], work.get("publisher", "")]
        source = ", ".join(part for part in source_parts if part)
        return {
            "title": title,
            "year": year or "",
            "authors": author_text or "",
            "source": source,
        }

    def _render_bibliography(self, entries: Sequence[BibliographyEntry]) -> str:
        if not entries:
            return ""
        lines = ["\\begin{thebibliography}{99}"]
        for entry in entries:
            lines.append(f"  \\bibitem{{{entry.key}}} {entry.render()}")
        lines.append("\\end{thebibliography}")
        return "\n".join(lines)


__all__ = ["BibliographyGenerator", "BibliographyResult"]
