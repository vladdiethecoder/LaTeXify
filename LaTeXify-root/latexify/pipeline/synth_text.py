from __future__ import annotations

"""Deterministic text synthesis helpers."""

from typing import Dict, List, Sequence, Tuple

from .synth_shared import (
    capabilities_from_text,
    collect_text_segments,
    sanitize_inline,
    slugify,
    title_from_question,
    user_flag_uncertain,
)


def _section_header(task_id: str, question: str) -> Tuple[str, str]:
    title = title_from_question(question, task_id)
    return title, slugify(title)


def _abstract_block(title: str, body_lines: Sequence[str], *, uncertain: bool) -> str:
    lines = ["\\begin{abstract}"]
    if body_lines:
        lines.extend(body_lines)
    else:
        lines.append(sanitize_inline(title))
    if uncertain:
        lines.append(r"\\todo{Verify OCR accuracy.}")
    lines.append("\\end{abstract}")
    return "\n".join(lines) + "\n"


def synthesize(bundle: Dict) -> Tuple[str, List[str]]:
    task_id = str(bundle.get("task_id", bundle.get("id", "task")))
    question = str(bundle.get("question", task_id))
    title, slug = _section_header(task_id, question)
    rubric_texts = collect_text_segments(bundle.get("rubric"))
    user_chunks = collect_text_segments(bundle.get("user_answer", {}).get("chunks"))
    uncertain = user_flag_uncertain(bundle)

    if title.lower().startswith("abstract"):
        body_lines = [sanitize_inline(chunk) for chunk in user_chunks] if user_chunks else []
        snippet = _abstract_block(title, body_lines, uncertain=uncertain)
        return snippet, capabilities_from_text(snippet)

    lines: List[str] = [
        f"\\section{{{sanitize_inline(title)}}}",
        f"\\label{{sec:{task_id}-{slug}}}",
    ]
    if rubric_texts:
        lines.append("% Rubric guidance")
        for note in rubric_texts:
            lines.append(f"\\textit{{{sanitize_inline(note)}}}")
    for chunk in user_chunks:
        lines.append(sanitize_inline(chunk))
    if uncertain:
        lines.append(r"\\todo{Verify OCR accuracy.}")
    snippet = "\n".join(lines) + "\n"
    return snippet, capabilities_from_text(snippet)


def synthesize_cli(bundle: Dict) -> str:
    task_id = str(bundle.get("task_id", bundle.get("id", "task")))
    question = str(bundle.get("question", task_id))
    header = f"\\section*{{Task {task_id}: {sanitize_inline(question)}}}"
    label = f"\\label{{sec:{task_id.lower()}}}"
    sections = [header, label]
    rubric_texts = collect_text_segments(bundle.get("assignment_rules")) or collect_text_segments(
        bundle.get("rubric")
    )
    if rubric_texts:
        sections.append("% Assignment guidance")
        for note in rubric_texts:
            sections.append(f"\\begin{{itemize}}\\item {sanitize_inline(note)}\\end{{itemize}}")
    user_chunks = collect_text_segments(bundle.get("user_answer", {}).get("chunks"))
    if user_chunks:
        sections.append("% User answer context")
        for chunk in user_chunks:
            sections.append(sanitize_inline(chunk))
    sections.extend(
        [
            "\\begin{align}",
            "  a + b &= c \\label{eq:example}\\\\",
            "  d &= e + f",
            "\\end{align}",
            "As a reference we will use \\SI{9.81}{\\meter\\per\\second\\squared}.",
            "\\begin{table}[h]",
            "\\centering",
            "\\begin{tabular}{ll}",
            "\\toprule",
            "Quantity & Value\\\\",
            "\\midrule",
            "Example & 1.0\\\\",
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Auto-generated reference table}",
            f"\\label{{tab:{task_id.lower()}}}",
            "\\end{table}",
        ]
    )
    if user_flag_uncertain(bundle):
        sections.append(r"\\todo{Verify OCR accuracy.}")
    return "\n\n".join(sections) + "\n"
