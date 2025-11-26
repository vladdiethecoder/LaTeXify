"""Specialist agents that generate LaTeX snippets per content type."""
from __future__ import annotations

import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

from PIL import Image

from ..core import common
from ..models.model_adapters import FlorenceAdapter, FlorenceConfig
from ..models.vlm_adapters import (
    BaseVLMAdapter,
    available_vlm_backends,
    get_vlm_adapter,
    resolve_vlm_backend,
)
from ..agents.figure_table_agent import FigureTableAgent
from .equation_normalizer import equation_normalizer
from .code_block_detector import wrap_code_blocks
from .graph_to_tikz import GraphToTikZGenerator
from .markdown_translator import MarkdownTranslator

LOGGER = logging.getLogger(__name__)
PLAIN_TEXT_ESCAPE_RE = re.compile(r"(?<!\\)([&%])")
MODELS_ROOT = Path(
    os.environ.get(
        "LATEXIFY_MODELS_ROOT",
        str(Path(__file__).resolve().parents[2] / "models"),
    )
).expanduser().resolve()
TABLE_VLM_BACKEND = os.environ.get("LATEXIFY_TABLE_VLM", "internvl").lower()
FIGURE_VLM_BACKEND = os.environ.get("LATEXIFY_FIGURE_VLM", TABLE_VLM_BACKEND).lower()
TABLE_PROMPT = (
    "You are a LaTeX assistant. Convert the table shown in the image into a LaTeX tabular "
    "environment using booktabs rules. Include column alignment, headers, and rows. "
    "Return only LaTeX."
)
FIGURE_PROMPT = (
    "Describe the figure in one concise sentence suitable for a LaTeX caption. "
    "Mention axes or key objects when relevant."
)
markdown_translator = MarkdownTranslator()
SEE_REF_RE = re.compile(r"(?i)\b(see|refer to)\s+(question|problem)\s+([0-9]+[A-Za-z]?)")
QUESTION_TOKEN_RE = re.compile(r"(?i)\b(question|problem)\s+([0-9]+[A-Za-z]?)")


def _question_slug(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"[^0-9a-z]+", "", value.lower())


def _resolve_cross_references(text: str) -> str:
    def _replacement(match: re.Match[str]) -> str:
        prefix = match.group(1)
        noun = match.group(2)
        label = _question_slug(match.group(3))
        if not label:
            return match.group(0)
        return f"{prefix} {noun}~\\ref{{q:{label}}}"

    updated = SEE_REF_RE.sub(_replacement, text)

    def _noun_replace(match: re.Match[str]) -> str:
        noun = match.group(1)
        label = _question_slug(match.group(2))
        if not label:
            return match.group(0)
        return f"{noun}~\\ref{{q:{label}}}"

    return QUESTION_TOKEN_RE.sub(_noun_replace, updated)


def _florence_model_dir() -> Path:
    return MODELS_ROOT / "ocr" / "florence-2-large"


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
        self.request("amsmath")
        self.request("amssymb")
        self.request("amsfonts")

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


def _escape_plain_text(text: str) -> str:
    return PLAIN_TEXT_ESCAPE_RE.sub(r"\\\1", text)


class VisionLanguageGenerator:
    """Caches Florence or pluggable VLM adapters for reusable prompts."""

    def __init__(self, prompt: str, backend: str, max_new_tokens: int = 512) -> None:
        self.prompt = prompt
        self.backend = backend or "internvl"
        self.max_new_tokens = max_new_tokens
        self._adapter: FlorenceAdapter | BaseVLMAdapter | None = None

    def _ensure_adapter(self):
        if self._adapter is not None:
            return self._adapter
        requested_backend = (self.backend or "internvl").lower()
        if requested_backend == "florence2":
            resolved_backend = "florence2"
        else:
            resolved_backend = resolve_vlm_backend(requested_backend)
        try:
            if resolved_backend == "florence2":
                config = FlorenceConfig(
                    model_dir=_florence_model_dir(),
                    task_prompt=self.prompt,
                    max_new_tokens=self.max_new_tokens,
                )
                self._adapter = FlorenceAdapter(config)
            elif resolved_backend in available_vlm_backends():
                self._adapter = get_vlm_adapter(
                    resolved_backend,
                    prompt=self.prompt,
                    max_new_tokens=self.max_new_tokens,
                    temperature=0.0,
                    top_p=0.1,
                )
            else:
                fallback = get_vlm_adapter(
                    "internvl",
                    prompt=self.prompt,
                    max_new_tokens=self.max_new_tokens,
                    temperature=0.0,
                    top_p=0.1,
                )
                LOGGER.warning(
                    "Unknown VLM backend '%s'; defaulting to %s.", resolved_backend, fallback.backend_name
                )
                self._adapter = fallback
        except Exception as exc:
            LOGGER.warning("VLM backend '%s' unavailable (%s); falling back to heuristics.", resolved_backend, exc)
            self._adapter = None
        return self._adapter

    def generate(self, image_path: Path) -> str | None:
        adapter = self._ensure_adapter()
        if adapter is None:
            return None
        try:
            if isinstance(adapter, BaseVLMAdapter):
                return adapter.describe(image_path, prompt=self.prompt)
            if hasattr(adapter, "prompt"):
                adapter.prompt = self.prompt
            return adapter.predict(image_path).strip()
        except Exception as exc:
            LOGGER.debug("VLM generation failed (%s): %s", image_path, exc)
            return None


table_vlm_generator = VisionLanguageGenerator(TABLE_PROMPT, TABLE_VLM_BACKEND, max_new_tokens=768)
figure_captioner = VisionLanguageGenerator(FIGURE_PROMPT, FIGURE_VLM_BACKEND, max_new_tokens=256)
figure_table_agent = FigureTableAgent()
graph_tikz_generator = GraphToTikZGenerator()
ENABLE_TIKZ_GRAPHS = os.environ.get("LATEXIFY_TIKZ_GRAPHS", "1") != "0"


def _normalize_paragraphs(text: str) -> str:
    markdown_processed = markdown_translator.convert(text)
    markdown_processed = wrap_code_blocks(markdown_processed)
    paragraphs = [line.strip() for line in markdown_processed.split("\n\n") if line.strip()]
    if not paragraphs:
        paragraphs = [markdown_processed.strip() or "[empty snippet]"]
    resolved = [_resolve_cross_references(paragraph) for paragraph in paragraphs]
    escaped = [_escape_plain_text(paragraph) for paragraph in resolved]
    return "\n\n".join(escaped)


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
    slug = _question_slug(number) or f"q{chunk.page:02d}"
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
            f"\\label{{q:{slug}}}",
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
    body = equation_normalizer.normalize(chunk.text).strip() or "[equation unavailable]"
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


def _split_row_cells(row: str) -> List[str]:
    if "|" in row:
        cells = [cell.strip() for cell in row.split("|")]
    elif "\t" in row:
        cells = [cell.strip() for cell in row.split("\t")]
    elif "," in row and len(row.split(",")) > 1:
        cells = [cell.strip() for cell in row.split(",")]
    else:
        cells = [cell.strip() for cell in re.split(r"\s{2,}", row)]
    return [cell for cell in cells if cell]


def _escape_table_cell(cell: str) -> str:
    replacements = {
        "&": "\\&",
        "%": "\\%",
        "#": "\\#",
    }
    sanitized = cell.strip()
    for needle, repl in replacements.items():
        sanitized = sanitized.replace(needle, repl)
    return sanitized


def _estimate_column_count(rows: List[str], metadata: Dict[str, object]) -> int:
    signature = metadata.get("table_signature") or {}
    columns = signature.get("columns")
    if isinstance(columns, int) and columns > 0:
        return columns
    if isinstance(columns, str) and columns.isdigit():
        return max(1, int(columns))
    counts = []
    for row in rows:
        cells = _split_row_cells(row)
        if cells:
            counts.append(len(cells))
    if not counts:
        return 1
    return max(max(counts), 1)


def _extract_region_crop(metadata: Dict[str, object]) -> Path | None:
    bbox = metadata.get("bbox")
    page_image = metadata.get("page_image")
    page_width = metadata.get("page_width_pt") or metadata.get("page_width")
    page_height = metadata.get("page_height_pt") or metadata.get("page_height")
    if not page_image or not bbox or page_width in (None, 0) or page_height in (None, 0):
        return None
    try:
        bbox_tuple = tuple(float(coord) for coord in bbox)
    except Exception:
        return None
    try:
        with Image.open(page_image) as image:
            width_px, height_px = image.size
            scale_x = width_px / max(float(page_width), 1e-3)
            scale_y = height_px / max(float(page_height), 1e-3)
            x0, y0, x1, y1 = bbox_tuple
            left = int(max(0, min(width_px, x0 * scale_x)))
            top = int(max(0, min(height_px, y0 * scale_y)))
            right = int(max(left + 1, min(width_px, x1 * scale_x)))
            bottom = int(max(top + 1, min(height_px, y1 * scale_y)))
            margin_x = max(2, int(0.01 * width_px))
            margin_y = max(2, int(0.01 * height_px))
            left = max(0, left - margin_x)
            right = min(width_px, right + margin_x)
            top = max(0, top - margin_y)
            bottom = min(height_px, bottom + margin_y)
            if left >= right or top >= bottom:
                return None
            crop = image.crop((left, top, right, bottom))
            temp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            crop.save(temp.name)
            temp.close()
            return Path(temp.name)
    except Exception as exc:
        LOGGER.debug("Failed to crop region image: %s", exc)
        return None


def _generate_table_from_vlm(chunk: common.Chunk) -> List[List[str]] | None:
    crop_path = _extract_region_crop(chunk.metadata or {})
    if crop_path is None:
        return None
    try:
        vlm_text = table_vlm_generator.generate(crop_path)
    finally:
        try:
            crop_path.unlink()
        except Exception:
            pass
    if not vlm_text:
        return None
    rows = []
    for raw_line in vlm_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        cells = _split_row_cells(line)
        if cells:
            rows.append(cells)
    return rows or None


def _generate_figure_caption(chunk: common.Chunk) -> str | None:
    metadata = chunk.metadata or {}
    chunk.metadata = metadata
    try:
        agent_caption, score = figure_table_agent.caption_from_chunk(chunk)
    except Exception as exc:  # pragma: no cover - depends on cv2 availability
        LOGGER.debug("Figure table agent caption fallback: %s", exc)
        agent_caption = None
        score = 0.0
    if agent_caption:
        metadata["figure_caption_score"] = score
        return agent_caption
    refs = metadata.get("image_refs") or chunk.images or []
    for ref in refs:
        try:
            path = Path(ref)
        except Exception:
            continue
        if not path.exists():
            continue
        caption = figure_captioner.generate(path)
        if caption:
            return caption
    crop_path = _extract_region_crop(chunk.metadata or {})
    if crop_path:
        try:
            caption = figure_captioner.generate(crop_path)
            if caption:
                return caption
        finally:
            try:
                crop_path.unlink()
            except Exception:
                pass
    return None


def _generate_tikz_plot(chunk: common.Chunk) -> str | None:
    crop_path = _extract_region_crop(chunk.metadata or {})
    if crop_path is None:
        return None
    try:
        tikz = graph_tikz_generator.generate(crop_path)
    finally:
        try:
            crop_path.unlink()
        except Exception:
            pass
    return tikz


def table_agent(
    chunk: common.Chunk,
    preamble: PreambleAgent,
    examples=None,
    context: Dict[str, object] | None = None,
) -> SpecialistResult:
    preamble.request("booktabs")
    metadata = chunk.metadata or {}
    rows = [row for row in chunk.text.splitlines() if row.strip()]
    cols = _estimate_column_count(rows, metadata)
    align = "".join(["c"] * cols)
    if examples:
        align_hint = _extract_tabular_align(examples[0].text)
        if align_hint:
            align = align_hint
        for pkg in examples[0].packages:
            preamble.request(pkg)
    context_comment = _context_comment(context)
    assessment = figure_table_agent.assess_table(chunk)
    width_value = min(0.98, max(0.65, assessment.recommended_width))
    header_parts = []
    if context_comment:
        header_parts.append(context_comment.strip())
    header_parts.append(_rag_comment(examples[0].doc_id) if examples else "% auto table")
    header_parts.append(
        f"% table-profile rows={assessment.rows} cols={assessment.columns} width={width_value:.2f}\\linewidth"
    )
    header = "\n".join(filter(None, header_parts))
    vlm_rows = _generate_table_from_vlm(chunk)
    if vlm_rows:
        cols = max(len(row) for row in vlm_rows)
        body_lines = []
        for row in vlm_rows:
            cells = [
                _escape_table_cell(cell)
                for cell in (row[:cols] + [""] * max(0, cols - len(row)))
            ]
            body_lines.append(" & ".join(cells) + " \\\\ ")
        body = "\n    ".join(body_lines)
        header = header or "% auto table"
        align = "".join(["c"] * max(cols, 1))
        latex = "\n".join(
            [
                header,
                "\\begin{table}[H]",
                "  \\centering",
                f"  \\resizebox{{{width_value:.2f}\\linewidth}}{{!}}{{%",
                f"    \\begin{{tabular}}{{{align}}}",
                "    \\toprule",
                f"    {body}",
                "    \\bottomrule",
                "    \\end{tabular}",
                "  }",
                "  \\caption{Auto-transcribed table}",
                "\\end{table}",
            ]
        )
        return SpecialistResult(latex=latex)
    body_lines: List[str] = []
    for row in rows:
        cells = _split_row_cells(row)
        if not cells:
            continue
        body_lines.append(" & ".join(cells[:cols]) + " \\\\")
    if not body_lines:
        body_lines.append(f"\\multicolumn{{{cols}}}{{c}}{{[table content unavailable]}} \\\\")
    body = "\n    ".join(body_lines)
    header = header or "% auto table"
    latex = "\n".join(
        [
            header,
            "\\begin{table}[H]",
            "  \\centering",
            f"  \\resizebox{{{width_value:.2f}\\linewidth}}{{!}}{{%",
            f"    \\begin{{tabular}}{{{align}}}",
            "    \\toprule",
            f"    {body}",
            "    \\bottomrule",
            "    \\end{tabular}",
            "  }",
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
    caption = (
        _generate_figure_caption(chunk)
        or chunk.metadata.get("figure_caption")
        or "Auto-captioned figure"
    )
    header = _context_comment(context) + (
        _rag_comment(examples[0].doc_id) if examples else "% figure placeholder"
    )
    tikz_snippet = None
    if ENABLE_TIKZ_GRAPHS:
        meta_text = " ".join(
            filter(
                None,
                [
                    chunk.metadata.get("figure_caption"),
                    chunk.metadata.get("label"),
                    chunk.text,
                ],
            )
        ).lower()
        if any(keyword in meta_text for keyword in ("graph", "plot", "curve", "chart", "diagram")):
            tikz_snippet = _generate_tikz_plot(chunk)
    if tikz_snippet:
        preamble.request("tikz")
        latex_body = "\n".join(
            [
                "\\begin{figure}[H]",
                "  \\centering",
                tikz_snippet.strip(),
                f"  \\caption{{{caption}}}",
                "\\end{figure}",
            ]
        )
        return SpecialistResult(latex=f"{header}\n{latex_body}")
    latex = f"{header}\n\\begin{{center}}\n\\textit{{{caption}}}\n\\end{{center}}"
    return SpecialistResult(latex=latex)


ROUTING_ALIASES = {
    "math": "equation",
    "formula": "equation",
    "eq": "equation",
    "display_equation": "equation",
    "img": "figure",
    "question_block": "question",
    "answer_block": "paragraph", # Map answer to paragraph for now, or add answer_agent
    "theorem_block": "paragraph", # Map to paragraph (will be formatted by latex content)
    "proof_block": "paragraph",
}
ROUTING_FIGURE_CONFIDENCE = float(os.environ.get("LATEXIFY_ROUTING_FIGURE_THRESHOLD", "0.45") or 0.45)


def _route_modality(block_type: str, chunk: common.Chunk) -> Dict[str, object]:
    """Centralized modality routing with configurable thresholds.

    This consolidates heuristic checks so downstream agents share the same decision.
    """
    metadata = chunk.metadata or {}
    region = str(metadata.get("region_type") or block_type or "text").lower()
    region = ROUTING_ALIASES.get(region, region)
    layout_conf = float(metadata.get("layout_confidence", 0.7) or 0.7)
    if region not in {"question", "table", "equation", "list", "figure"}:
        images = getattr(chunk, "images", None) or []
        if images and layout_conf < ROUTING_FIGURE_CONFIDENCE:
            region = "figure"
        elif metadata.get("table_signature"):
            region = "table"
        elif "equation" in chunk.text.lower() or "formula" in chunk.text.lower():
            region = "equation"
        elif metadata.get("bulleted") or metadata.get("list"):
            region = "list"
        else:
            region = "text"
    return {"region": region, "layout_confidence": layout_conf}


def dispatch_specialist(
    block_type: str,
    chunk: common.Chunk,
    preamble: PreambleAgent,
    examples: Sequence[object] | None = None,
    *,
    context: Dict[str, object] | None = None,
) -> SpecialistResult:
    routing = _route_modality(block_type, chunk)
    region = routing["region"]
    if region == "question":
        result = question_agent(chunk, preamble, examples, context)
    elif region == "table":
        result = table_agent(chunk, preamble, examples, context)
    elif region == "equation":
        result = equation_agent(chunk, preamble, examples, context)
    elif region == "list":
        result = list_agent(chunk, examples, context)
    elif region == "figure":
        result = figure_agent(chunk, examples, context)
    else:
        result = paragraph_agent(chunk, examples, context)
    result.notes["routing"] = routing
    return result


__all__ = ["PreambleAgent", "dispatch_specialist", "SpecialistResult"]
