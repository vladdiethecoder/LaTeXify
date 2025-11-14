"""Specialist agents that generate LaTeX snippets per content type."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from ..core import common


@dataclass
class SpecialistResult:
    latex: str
    notes: Dict[str, object] = field(default_factory=dict)


class PreambleAgent:
    """Collects required packages from specialist agents."""

    def __init__(self) -> None:
        # maintain insertion order
        self._packages: Dict[str, str | None] = {}
        self.request("graphicx")
        self.request("geometry", options="margin=1in")
        self.request("float")

    def request(self, package: str, options: str | None = None) -> None:
        existing = self._packages.get(package)
        if existing is None:
            self._packages[package] = options
            return
        if options and existing != options:
            # prefer explicit options; overwrite default placeholder (e.g., geometry)
            self._packages[package] = options

    def packages(self) -> List[Dict[str, str | None]]:
        ordered = list(self._packages.items())
        # ensure hyperref is loaded last if requested
        ordered_sorted: List[tuple[str, str | None]] = []
        hyperref_entry = None
        for pkg, opts in ordered:
            if pkg == "hyperref":
                hyperref_entry = (pkg, opts)
            else:
                ordered_sorted.append((pkg, opts))
        if hyperref_entry:
            ordered_sorted.append(hyperref_entry)
        return [{"package": pkg, "options": opts} for pkg, opts in ordered_sorted]


def _normalize_paragraphs(text: str) -> str:
    paragraphs = [line.strip() for line in text.split("\n\n") if line.strip()]
    if not paragraphs:
        paragraphs = [text.strip() or "[empty snippet]"]
    return "\n\n".join(paragraphs)


def _rag_comment(example_label: str) -> str:
    return f"% RAG reference: {example_label}"


def _context_comment(context: Dict[str, object] | None) -> str:
    if not context:
        return ""
    section = context.get("section_title")
    summary = context.get("section_summary")
    parts = []
    if section:
        parts.append(f"section={section}")
    if summary:
        preview = str(summary)
        if len(preview) > 80:
            preview = preview[:77] + "..."
        parts.append(f"summary={preview}")
    if not parts:
        return ""
    return "% context: " + " | ".join(parts) + "\n"


def paragraph_agent(chunk: common.Chunk, examples=None, context: Dict[str, object] | None = None) -> SpecialistResult:
    latex = _normalize_paragraphs(chunk.text)
    header = _context_comment(context)
    if examples:
        latex = "\n".join([_rag_comment(examples[0].doc_id), latex])
    return SpecialistResult(latex=header + latex)


def question_agent(
    chunk: common.Chunk,
    preamble: PreambleAgent,
    examples=None,
    context: Dict[str, object] | None = None,
) -> SpecialistResult:
    preamble.request("tcolorbox")
    preamble.request("enumitem")
    meta = chunk.metadata or {}
    label = meta.get("question_label") or meta.get("header_label") or chunk.metadata.get("label")
    number = label or f"{chunk.page:02d}"
    body = _normalize_paragraphs(chunk.text)
    header = _context_comment(context)
    prefix_lines = [header.strip()] if header else []
    if examples:
        prefix_lines.append(_rag_comment(examples[0].doc_id))
    prefix = "\n".join([line for line in prefix_lines if line])
    latex = "\n".join(
        [
            prefix if prefix else "% question block",
            f"\\begin{{question}}{{{number}}}",
            body,
            "\\end{question}",
        ]
    )
    return SpecialistResult(latex=latex)


def equation_agent(
    chunk: common.Chunk,
    preamble: PreambleAgent,
    examples=None,
    context: Dict[str, object] | None = None,
) -> SpecialistResult:
    preamble.request("amsmath")
    env = "equation"
    if examples:
        sample = examples[0].text
        if "\\begin{align" in sample:
            env = "align"
        for pkg in examples[0].packages:
            preamble.request(pkg)
    body = chunk.text.strip() or "[equation unavailable]"
    header = (_context_comment(context) + (_rag_comment(examples[0].doc_id) if examples else "% equation snippet"))
    latex = "\n".join(
        [
            header.strip() if header else "% equation snippet",
            f"\\begin{{{env}}}",
            body,
            f"\\end{{{env}}}",
        ]
    )
    return SpecialistResult(latex=latex)


def _extract_tabular_align(example_text: str) -> str | None:
    match = re.search(r"\\begin\{tabular\}\{([^}]+)\}", example_text)
    if match:
        return match.group(1)
    return None


def table_agent(
    chunk: common.Chunk,
    preamble: PreambleAgent,
    examples=None,
    context: Dict[str, object] | None = None,
) -> SpecialistResult:
    preamble.request("booktabs")
    metadata = chunk.metadata or {}
    signature = metadata.get("table_signature") or {}
    rows = [row for row in chunk.text.splitlines() if row.strip()]
    cols = signature.get("columns") or max((row.count("|") for row in rows), default=2)
    cols = max(1, cols)
    align = "".join(["c"] * cols)
    if examples:
        align_hint = _extract_tabular_align(examples[0].text)
        if align_hint:
            align = align_hint
        for pkg in examples[0].packages:
            preamble.request(pkg)
    body_lines: List[str] = []
    for row in rows:
        if "|" in row:
            cells = [cell.strip() for cell in row.split("|") if cell.strip()]
        else:
            cells = [cell.strip() for cell in row.split() if cell.strip()]
        if not cells:
            continue
        body_lines.append(" & ".join(cells[:cols]) + " \\\\")
    if not body_lines:
        body_lines.append(f"\\multicolumn{{{cols}}}{{c}}{{[table content unavailable]}} \\\\")
    body = "\n    ".join(body_lines)
    context_comment = _context_comment(context)
    header = (context_comment + (_rag_comment(examples[0].doc_id) if examples else "% auto table")).strip()
    latex = "\n".join(
        [
            header,
            "\\begin{table}[H]",
            "  \\centering",
            f"  \\begin{{tabular}}{{{align}}}",
            "    \\toprule",
            f"    {body}",
            "    \\bottomrule",
            "  \\end{tabular}",
            "  \\caption{Auto-transcribed table}",
            "\\end{table}",
        ]
    )
    return SpecialistResult(latex=latex)


def list_agent(chunk: common.Chunk, examples=None, context: Dict[str, object] | None = None) -> SpecialistResult:
    lines = [line for line in chunk.text.splitlines() if line.strip()]
    first = lines[0] if lines else ""
    ordered = first.lstrip().startswith(tuple("0123456789"))
    env = "enumerate" if ordered else "itemize"
    items = []
    for line in chunk.text.splitlines():
        stripped = line.strip("-*• \t")
        if not stripped:
            continue
        items.append(f"  \\item {stripped}")
    if not items:
        items.append("  \\item [list content unavailable]")
    header = []
    context_comment = _context_comment(context)
    if context_comment:
        header.append(context_comment.strip())
    header.append(_rag_comment(examples[0].doc_id) if examples else "% auto list")
    latex = "\n".join(["\n".join(header), f"\\begin{{{env}}}", *items, f"\\end{{{env}}}"])
    return SpecialistResult(latex=latex)


def figure_agent(chunk: common.Chunk, examples=None, context: Dict[str, object] | None = None) -> SpecialistResult:
    caption = chunk.metadata.get("figure_caption") or "Auto-captioned figure"
    header = _context_comment(context) + (
        _rag_comment(examples[0].doc_id) if examples else "% figure placeholder"
    )
    latex = f"{header}\n\\begin{{center}}\n\\textit{{{caption}}}\n\\end{{center}}"
    return SpecialistResult(latex=latex)


def dispatch_specialist(
    block_type: str,
    chunk: common.Chunk,
    preamble: PreambleAgent,
    examples: Sequence[object] | None = None,
    *,
    context: Dict[str, object] | None = None,
) -> SpecialistResult:
    region = chunk.metadata.get("region_type", "text")
    if block_type == "question" or region == "question":
        return question_agent(chunk, preamble, examples, context)
    if block_type == "table" or region == "table":
        return table_agent(chunk, preamble, examples, context)
    if block_type == "equation" or region == "formula":
        return equation_agent(chunk, preamble, examples, context)
    if block_type == "list" or region == "list":
        return list_agent(chunk, examples, context)
    if block_type == "figure" or region == "figure":
        return figure_agent(chunk, examples, context)
    return paragraph_agent(chunk, examples, context)


__all__ = ["PreambleAgent", "dispatch_specialist", "SpecialistResult"]
