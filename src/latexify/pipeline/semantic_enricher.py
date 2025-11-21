"""Annotate snippets with domain-specific LaTeX environments."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from ..core import common


THEOREM_LABEL_RE = re.compile(r"^(theorem|lemma|definition|corollary|proposition)\b", re.IGNORECASE)
ALGORITHM_RE = re.compile(r"\b(algorithm|procedure)\b", re.IGNORECASE)
PROOF_RE = re.compile(r"^(proof|sketch)\b", re.IGNORECASE)


@dataclass
class EnrichmentReport:
    updated: int
    entries: List[Dict[str, str]]
    domain: str
    mode: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "updated": self.updated,
            "entries": self.entries,
            "domain": self.domain,
            "mode": self.mode,
        }


class SemanticEnricher:
    """Attach theorem/proof/algorithm environments based on content analysis."""

    def __init__(
        self,
        domain_profile: Dict[str, object] | None,
        quality_profile: Dict[str, object] | None,
    ) -> None:
        self.domain = (domain_profile or {}).get("domain", "general")
        mode = (quality_profile or {}).get("processing_mode", "balanced")
        self.conservative = mode == "conservative"
        self.records: List[Dict[str, str]] = []

    def enrich(
        self,
        plan_path: Path,
        chunks_path: Path,
        snippets_path: Path,
    ) -> EnrichmentReport:
        plan_blocks = common.load_plan(plan_path)
        block_lookup = {block.chunk_id: block for block in plan_blocks}
        chunk_map = {chunk.chunk_id: chunk for chunk in common.load_chunks(chunks_path)}
        snippets = list(common.load_snippets(snippets_path))
        updated = 0
        for snippet in snippets:
            block = block_lookup.get(snippet.chunk_id)
            chunk = chunk_map.get(snippet.chunk_id)
            if not block or not chunk:
                continue
            new_body, label = self._enrich_snippet(chunk, block, snippet.latex)
            if new_body != snippet.latex:
                snippet.latex = new_body
                snippet.notes.setdefault("semantic_enrichment", label)
                self.records.append({"chunk_id": snippet.chunk_id, "label": label})
                updated += 1
        if updated:
            common.save_snippets(snippets, snippets_path)
        return EnrichmentReport(
            updated=updated,
            entries=self.records[:32],
            domain=self.domain,
            mode="conservative" if self.conservative else "dynamic",
        )

    # ------------------------------- helpers ------------------------------- #
    def _enrich_snippet(
        self,
        chunk: common.Chunk,
        block: common.PlanBlock,
        latex: str,
    ) -> tuple[str, str]:
        text = chunk.text.strip()
        metadata = chunk.metadata or {}
        if self._should_wrap_algorithm(text):
            return self._wrap_algorithm(text, latex), "algorithm"
        if self._should_wrap_theorem(text):
            env = self._choose_theorem_env(text)
            return self._wrap_environment(env, latex, text), env
        if self._should_wrap_proof(text, metadata):
            return self._wrap_environment("proof", latex, text, close_with_qed=True), "proof"
        return latex, ""

    def _should_wrap_theorem(self, text: str) -> bool:
        if self.domain not in {"mathematics", "physics"}:
            return False
        match = THEOREM_LABEL_RE.match(text)
        if not match:
            return False
        if self.conservative and not text.splitlines()[0].strip().endswith("."):
            return False
        return True

    def _choose_theorem_env(self, text: str) -> str:
        match = THEOREM_LABEL_RE.match(text)
        if not match:
            return "theorem"
        keyword = match.group(1).lower()
        if keyword == "definition":
            return "definition"
        if keyword == "lemma":
            return "lemma"
        if keyword == "corollary":
            return "corollary"
        return "theorem"

    def _should_wrap_proof(self, text: str, metadata: Dict[str, object]) -> bool:
        if self.domain not in {"mathematics", "physics"}:
            return False
        role = str(metadata.get("math_role", ""))
        if role.startswith("proof"):
            return True
        return bool(PROOF_RE.match(text))

    def _should_wrap_algorithm(self, text: str) -> bool:
        if self.domain not in {"computer_science", "engineering"}:
            return False
        if not ALGORITHM_RE.search(text):
            return False
        if self.conservative and "input:" not in text.lower():
            return False
        return True

    def _wrap_environment(
        self,
        env: str,
        body: str,
        source_text: str,
        close_with_qed: bool = False,
    ) -> str:
        trimmed = body.strip()
        lines = [f"\\begin{{{env}}}"]
        lines.append(trimmed)
        if close_with_qed and "\\qedhere" not in trimmed:
            lines.append("\\qedhere")
        lines.append(f"\\end{{{env}}}")
        return "\n".join(lines)

    def _wrap_algorithm(self, text: str, body: str) -> str:
        caption = self._algorithm_caption(text)
        lines = [
            "\\begin{algorithm}[H]",
            f"\\caption{{{caption}}}",
            "\\begin{algorithmic}",
            body.strip(),
            "\\end{algorithmic}",
            "\\end{algorithm}",
        ]
        return "\n".join(lines)

    def _algorithm_caption(self, text: str) -> str:
        first_line = text.splitlines()[0].strip()
        if len(first_line) > 80:
            first_line = first_line[:77] + "..."
        return first_line or "Auto-generated procedure"


__all__ = ["SemanticEnricher", "EnrichmentReport"]
