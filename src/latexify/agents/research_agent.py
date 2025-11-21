"""DuckDuckGo-backed ResearchAgent with offline fallback."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Iterable, List, Protocol

from .graph_state import GraphState

LOGGER = logging.getLogger(__name__)


class SearchBackend(Protocol):
    def fetch(self, query: str, max_results: int) -> List["ResearchSnippet"]:
        ...


@dataclass
class ResearchSnippet:
    source: str
    content: str
    url: str | None = None


class DuckDuckGoBackend:
    """Zero-cost backend that uses duckduckgo-search (if installed)."""

    def __init__(self) -> None:
        try:
            from duckduckgo_search import DDGS  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("duckduckgo-search is not installed") from exc
        self._ddgs = DDGS()

    def fetch(self, query: str, max_results: int) -> List[ResearchSnippet]:
        results: List[ResearchSnippet] = []
        try:
            for item in self._ddgs.text(query, max_results=max_results):
                title = item.get("title") or "result"
                body = item.get("body") or ""
                url = item.get("href")
                snippet = ResearchSnippet(
                    source=title,
                    content=body.strip() or f"See {url}",
                    url=url,
                )
                results.append(snippet)
        except Exception as exc:  # pragma: no cover - network errors
            LOGGER.warning("DuckDuckGo search failed (%s)", exc)
        return results


class StubBackend:
    def fetch(self, query: str, max_results: int) -> List[ResearchSnippet]:
        hint = query.splitlines()[0][:120] if query.strip() else "layout"
        return [
            ResearchSnippet(
                source="stub",
                content=f"\\textit{{Reference hint based on '{hint}'}}",
            )
        ]


class ResearchAgent:
    """Free research agent that can hit DuckDuckGo or remain offline."""

    def __init__(self, backend: SearchBackend | None = None, max_results: int = 3) -> None:
        self.max_results = max_results
        self.backend = backend or self._default_backend()

    def _default_backend(self) -> SearchBackend:
        if os.environ.get("RESEARCH_AGENT_OFFLINE", "0") == "1":
            LOGGER.info("[research] Offline mode enabled; using stub backend.")
            return StubBackend()
        try:
            return DuckDuckGoBackend()
        except RuntimeError as exc:
            LOGGER.warning("[research] %s; falling back to stub backend", exc)
            return StubBackend()

    def search(self, query: str) -> List[ResearchSnippet]:
        return self.backend.fetch(query, self.max_results)

    def augment(self, state: GraphState) -> GraphState:
        snippets = self.search(state.content)
        for snippet in snippets:
            rendered = snippet.content
            if snippet.url:
                rendered += f" (source: {snippet.url})"
            state.research_snippets.append(rendered)
        state.mark_stage("research", notes=str(len(snippets)))
        state.record_metrics(research_snippets=len(snippets))
        state.log(f"research: added {len(snippets)} snippet(s)")
        return state


__all__ = ["ResearchAgent", "ResearchSnippet"]
