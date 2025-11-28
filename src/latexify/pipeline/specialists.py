"""Specialist agents that generate LaTeX snippets per content type."""
from __future__ import annotations

import logging
import os
import re
import tempfile
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Any

from PIL import Image

from ..core import common
from ..models.model_adapters import FlorenceAdapter, FlorenceConfig
from ..models.vlm_adapters import (
    BaseVLMAdapter,
    available_vlm_backends,
    get_vlm_adapter,
    resolve_vlm_backend,
)
from ..models.vllm_client import get_vllm_client
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


def _question_slug(value: str | None) -> str:
    """
    Generate a clean alphanumeric slug from a question label/number.
    This is only used to build LaTeX labels, not to heuristically
    classify or route content.
    """
    if not value:
        return ""
    return re.sub(r"[^0-9a-z]", "", str(value).lower())


class SemanticRouter:
    """Uses LLM to route chunk to appropriate specialist agent."""
    def __init__(self):
        self._client = get_vllm_client()
        self._system_prompt = """You are a semantic router for a LaTeX conversion pipeline.
Analyze the input text chunk and classify it into one of the following categories:
- question: A distinct exam/homework question block (e.g. "Question 1", "Problem 3").
- answer: A distinct answer block, solution, or marking rubric entry.
- table: A tabular structure or matrix that should be formatted as a table.
- equation: A mathematical block (single or multiline).
- list: A bulleted or numbered list.
- figure: A caption or reference to an image/figure.
- text: General prose, paragraphs, or section headers.

Respond ONLY with a JSON object: {"category": "...", "confidence": 0.0-1.0}
"""

    def route(self, chunk: common.Chunk, block_type_hint: str) -> str:
        if not self._client:
            return block_type_hint or "text"
            
        prompt = f"Block Type Hint: {block_type_hint}\n\nContent:\n{chunk.text[:1000]}"
        try:
            response = self._client.generate(self._system_prompt + "\nInput:\n" + prompt, max_tokens=64)
            # Simple JSON extraction
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                return data.get("category", "text")
        except Exception as e:
            LOGGER.warning(f"Semantic router failed: {e}")
            
        return block_type_hint or "text"

semantic_router = SemanticRouter()


def _normalize_paragraphs(text: str) -> str:
    markdown_processed = markdown_translator.convert(text)
    markdown_processed = wrap_code_blocks(markdown_processed)
    paragraphs = [line.strip() for line in markdown_processed.split("\n\n") if line.strip()]
    if not paragraphs:
        paragraphs = [markdown_processed.strip() or "[empty snippet]"]
    escaped = [_escape_plain_text(paragraph) for paragraph in paragraphs]
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


def _extract_question_metadata(chunk: common.Chunk) -> Dict[str, str]:
    """
    Ask the LLM to parse question structure instead of regex heuristics.
    Returns a dict with label, points, topic, and body keys.
    """
    client = get_vllm_client()
    defaults = {
        "label": str(chunk.page + 1),
        "points": "",
        "topic": "",
        "body": chunk.text,
    }
    if not client:
        return defaults

    prompt = (
        "You are structuring an exam question for LaTeX.\n"
        "Extract the logical metadata and return JSON only.\n"
        'Fields: \"label\" (e.g. \"1\", \"1a\"), \"points\" (e.g. \"6pts\" or \"\"), '
        '\"topic\" (short topic name), and \"body\" (question text without leading labels).\n\n'
        f"Question:\n{chunk.text[:1200]}"
    )
    try:
        resp = client.generate(prompt, max_tokens=256)
        match = re.search(r"\{.*\}", resp, re.DOTALL)
        if not match:
            return defaults
        data = json.loads(match.group(0))
        return {
            "label": str(data.get("label") or defaults["label"]),
            "points": str(data.get("points") or "").strip(),
            "topic": str(data.get("topic") or "").strip(),
            "body": str(data.get("body") or defaults["body"]),
        }
    except Exception as exc:  # pragma: no cover - runtime guard
        LOGGER.warning("Question metadata extraction failed: %s", exc)
        return defaults


def question_agent(
    chunk: common.Chunk,
    preamble: PreambleAgent,
    examples=None,
    context: Dict[str, object] | None = None,
) -> SpecialistResult:
    preamble.request("tcolorbox")
    preamble.request("enumitem")
    meta = _extract_question_metadata(chunk)

    # Build stable LaTeX label for cross-references
    raw_label = meta.get("label") or str(chunk.page + 1)
    slug = _question_slug(raw_label)
    unique_suffix = (chunk.chunk_id or "gen")[-4:]
    label_def = f"q:{slug or raw_label}:{unique_suffix}"

    points = meta.get("points") or ""
    topic = meta.get("topic") or f"Question {raw_label}"
    body_source = meta.get("body") or chunk.text
    body = _normalize_paragraphs(body_source)

    header = _context_comment(context)
    prefix_lines = [header.strip()] if header else []
    if examples:
        prefix_lines.append(_rag_comment(examples[0].doc_id))
    prefix = "\n".join([line for line in prefix_lines if line])

    problem_opts = f"[{points}]" if points else ""
    latex = "\n".join(
        [
            prefix if prefix else "% question block",
            f"\\begin{{problem}}{problem_opts}{{{topic}}}",
            f"\\label{{{label_def}}}",
            body,
            "\\end{problem}",
        ]
    )
    return SpecialistResult(latex=latex)


def answer_agent(
    chunk: common.Chunk,
    preamble: PreambleAgent,
    examples=None,
    context: Dict[str, object] | None = None
) -> SpecialistResult:
    preamble.request("tcolorbox")
    preamble.request("enumitem")

    header = _context_comment(context)
    rag_prefix = _rag_comment(examples[0].doc_id) if examples else ""

    client = get_vllm_client()
    body = _normalize_paragraphs(chunk.text)
    if client:
        prompt = (
            "You are formatting an answer or grading rubric for a professional exam textbook.\n"
            "Rewrite the content as LaTeX suitable to be placed INSIDE an 'answer' tcolorbox environment.\n"
            "- Use bullet lists or enumerated steps when appropriate.\n"
            "- Preserve math notation but convert it to LaTeX.\n"
            "- Do NOT include \\begin{answer} or \\end{answer}.\n\n"
            f"Answer text:\n{chunk.text[:2000]}"
        )
        try:
            raw = client.generate(prompt, max_tokens=512)
            stripped = raw.strip()
            # Guard against the model echoing the environment
            if "\\begin{answer}" in stripped:
                stripped = stripped.replace("\\begin{answer}", "").replace("\\end{answer}", "").strip()
            body = stripped or body
        except Exception as exc:  # pragma: no cover - runtime guard
            LOGGER.warning("Answer agent LLM failed, falling back to normalized text: %s", exc)

    lines = [line for line in [header.strip(), rag_prefix.strip()] if line]
    prefix = "\n".join(lines)
    if prefix:
        prefix += "\n"

    latex = "".join(
        [
            prefix,
            "\\begin{answer}\n",
            body,
            "\n\\end{answer}",
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
    # Delegate completely to equation_normalizer which now uses LLM
    body = equation_normalizer.normalize(chunk.text).strip() or "[equation unavailable]"
    header = _context_comment(context)
    latex = f"{header.strip()}\n{body}" if header else body
    return SpecialistResult(latex=latex)


def _extract_tabular_align(example_text: str) -> str | None:
    match = re.search(r"\\begin\{tabular\}\{([^}]+)\}", example_text)
    if match:
        return match.group(1)
    return None


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


def _generate_table_from_vlm(chunk: common.Chunk) -> str | None:
    # UPDATED: Returns raw latex string from VLM, not rows
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
    return vlm_text


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
    
    # Try VLM first
    vlm_latex = _generate_table_from_vlm(chunk)
    
    if not vlm_latex:
        # Fallback to LLM text-to-table
        client = get_vllm_client()
        if client:
            prompt = f"Convert this text to a LaTeX table:\n{chunk.text}"
            try:
                vlm_latex = client.generate(prompt, max_tokens=512)
            except:
                pass
                
    if not vlm_latex:
        vlm_latex = "\\begin{table}[H]\\centering [Table generation failed] \\end{table}"

    # Ensure environment wrapper
    if "\\begin{table" not in vlm_latex:
        vlm_latex = f"\\begin{{table}}[H]\n\\centering\n{vlm_latex}\n\\caption{{Generated Table}}\n\\end{{table}}"

    context_comment = _context_comment(context) or ""
    rag_comment = _rag_comment(examples[0].doc_id) + "\n" if examples else ""
    prefix = context_comment + rag_comment
    return SpecialistResult(latex=f"{prefix}{vlm_latex}")


def list_agent(chunk: common.Chunk, examples=None, context: Dict[str, object] | None = None) -> SpecialistResult:
    # Use LLM for list formatting to avoid regex-heavy heuristics.
    client = get_vllm_client()
    if client:
        prompt = (
            "Format the following text as a LaTeX list.\n"
            "Choose itemize or enumerate based on the content structure.\n"
            "Return only the LaTeX list environment.\n\n"
            f"{chunk.text}"
        )
        try:
            latex = client.generate(prompt, max_tokens=512)
            return SpecialistResult(latex=latex.strip() or _normalize_paragraphs(chunk.text))
        except Exception as exc:  # pragma: no cover - runtime guard
            LOGGER.warning("List agent LLM failed, falling back to plain text: %s", exc)

    # Fallback: treat as plain paragraph if LLM is unavailable.
    return SpecialistResult(latex=_normalize_paragraphs(chunk.text))


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


def dispatch_specialist(
    block_type: str,
    chunk: common.Chunk,
    preamble: PreambleAgent,
    examples: Sequence[object] | None = None,
    *,
    context: Dict[str, object] | None = None,
) -> SpecialistResult:
    # ZERO HEURISTICS: Use Semantic Router
    region = semantic_router.route(chunk, block_type)
    
    if region == "question":
        result = question_agent(chunk, preamble, examples, context)
    elif region == "answer":
        result = answer_agent(chunk, preamble, examples, context)
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
        
    result.notes["routing"] = region
    return result


__all__ = ["PreambleAgent", "dispatch_specialist", "SpecialistResult"]
