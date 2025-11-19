"""Assembly stage that creates the final LaTeX document and optional PDF."""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from ..core import common
from ..core.hierarchical_schema import ReferenceIndex
from ..core.sanitizer import sanitize_unicode_to_latex
from ..agents.figure_table_agent import FigureRenderPlan, FigureTableAgent
from ..agents.refinement_agent import RefinementAgent
from .bibliography_utils import BibliographyGenerator, BibliographyResult
from .citation_detector import CitationDetector, CitationReport
from .consistency_utils import MathConsistencyValidator
from .cross_reference_resolver import resolve_references
from .error_repair import LaTeXErrorRepair
from .hierarchical_analyzer import analyze_plan
from .math_support import MathEnvironmentDetector
from .latex_repair_agent import KimiK2LatexRepair
from .robust_compilation import run_robust_compilation
from .symbol_normalizer import SymbolNormalizer
from .template_engine import LaTeXTemplatingEngine
from .style_detector import DocumentStyle, StyleDetector
from .typography_engine import TypographyEngine

LOGGER = logging.getLogger(__name__)

POSTAMBLE = "\\end{document}\n"


LATEX_ESCAPES = {
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}

_LAST_COMPILATION_METRICS: Dict[str, object] | None = None


def escape(text: str) -> str:
    if not text:
        return ""
    fragments: List[str] = []
    for char in text:
        if char == "\\":
            fragments.append(r"\textbackslash{}")
        elif char == "\n":
            fragments.append(r"\\ ")
        elif char in LATEX_ESCAPES:
            fragments.append(LATEX_ESCAPES[char])
        else:
            fragments.append(char)
    return "".join(fragments)


def _section_command(level: str | None, document_class: str) -> str:
    normalized = (document_class or "article").lower()
    if level == "part":
        return "\\part"
    if level == "chapter":
        return "\\chapter" if normalized in {"book", "report", "memoir"} else "\\section"
    if level == "subsection":
        return "\\subsection"
    if level == "subsubsection":
        return "\\subsubsection"
    return "\\section"


TEMPLATE_PREFIXES = (
    "style exemplar",
    "content-type",
    "source text",
    "baseline snippet",
    "respond with",
    "<latex>",
    "<<<source",
    "high-quality exemplars",
)

EQUATION_BLOCK_RE = re.compile(
    r"\\begin\{equation\*?\}([\s\S]*?)\\end\{equation\*?\}", re.IGNORECASE
)


def _looks_like_plain_text(payload: str) -> bool:
    stripped = payload.strip()
    if not stripped:
        return False
    if "```" in stripped or stripped.startswith("%"):
        return True
    letter_count = sum(1 for char in stripped if char.isalpha())
    backslash_count = stripped.count("\\")
    math_symbols = sum(1 for char in stripped if char in "=+-/*_^")
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if len(lines) >= 3:
        alpha_lines = sum(1 for line in lines if sum(1 for ch in line if ch.isalpha()) >= 4)
        colon_hits = sum(1 for line in lines if ":" in line)
        if alpha_lines >= 2 and colon_hits:
            return True
    return letter_count > 20 and backslash_count <= 1 and math_symbols < letter_count / 4


def _demote_plain_text_equations(snippet: str) -> str:
    def _replacement(match: re.Match[str]) -> str:
        body = match.group(1)
        if not _looks_like_plain_text(body):
            return match.group(0)
        text = " ".join(line.strip() for line in body.splitlines() if line.strip())
        if not text:
            return ""
        return f"\\par {escape(text)}\\par\n"

    return EQUATION_BLOCK_RE.sub(_replacement, snippet)


def _sanitize_snippet(snippet: str) -> str:
    cleaned_lines: List[str] = []
    skip_block = False
    for raw_line in snippet.splitlines():
        line = raw_line.strip()
        lower = line.lower()
        if lower.startswith("<<<source"):
            skip_block = True
            continue
        if skip_block and (line == "SOURCE" or line == ">>>"):
            skip_block = False
            continue
        if skip_block:
            continue
        if any(lower.startswith(prefix) for prefix in TEMPLATE_PREFIXES):
            continue
        if line in {"```latex", "```"}:
            continue
        cleaned_lines.append(raw_line)
    cleaned = "\n".join(cleaned_lines).strip()
    return _demote_plain_text_equations(cleaned)


def _copy_block_images(block: common.PlanBlock, assets_dir: Path, used_assets: set[str]) -> List[str]:
    paths: List[str] = []
    assets_dir.mkdir(parents=True, exist_ok=True)
    for image_path in block.images:
        source = Path(image_path)
        if not source.exists():
            LOGGER.debug("Image missing at %s", image_path)
            continue
        dest = assets_dir / source.name
        if dest.name not in used_assets:
            if source.resolve() != dest.resolve():
                shutil.copy2(source, dest)
            used_assets.add(dest.name)
        relative = dest.relative_to(dest.parent.parent)
        paths.append(str(relative))
    return paths


def render_figure(
    block: common.PlanBlock,
    assets_dir: Path,
    used_assets: set[str],
    references: ReferenceIndex | None = None,
    figure_plan: FigureRenderPlan | None = None,
) -> str:
    latex_blocks: List[str] = []
    figure_paths = _copy_block_images(block, assets_dir, used_assets)
    if not figure_paths:
        return ""

    def _label() -> str | None:
        if figure_plan and figure_plan.label_hint:
            return figure_plan.label_hint
        if not references:
            return (block.metadata or {}).get("resolved_label")
        return references.label_for_block(block.block_id) or (block.metadata or {}).get("resolved_label")

    def _caption() -> str:
        caption_text = figure_plan.caption if figure_plan and figure_plan.caption else block.label
        return escape(caption_text or "Auto-captioned figure")

    def _float_spec() -> str:
        if figure_plan and figure_plan.float_spec:
            return figure_plan.float_spec
        return "H"

    def _quality_comments() -> List[str]:
        comments: List[str] = []
        if figure_plan and figure_plan.classification:
            comments.append(f"% figure-type: {figure_plan.classification}")
        if figure_plan and figure_plan.caption_score:
            breakdown = ", ".join(f"{key}={value:.2f}" for key, value in figure_plan.quality_breakdown.items())
            comments.append(f"% figure-caption-score {figure_plan.caption_score:.2f} ({breakdown})")
        return comments

    def _render_subfigures() -> str:
        comments = _quality_comments()
        label = _label()
        lines = comments + [f"\\begin{{figure}}[{_float_spec()}]", "  \\centering"]
        for directive in figure_plan.subfigures:
            source_idx = min(directive.image_index, len(figure_paths) - 1)
            source = figure_paths[source_idx]
            trim_values = " ".join(f"{value}pt" for value in directive.trim)
            lines.extend(
                [
                    f"  \\begin{{subfigure}}{{{directive.width_ratio:.2f}\\linewidth}}",
                    f"    \\includegraphics[width=\\linewidth,clip,trim={trim_values}]{{{source}}}",
                    f"    \\caption{{{escape(directive.caption)}}}",
                    "  \\end{subfigure}",
                ]
            )
        lines.append(f"  \\caption{{{_caption()}}}")
        if label:
            lines.append(f"  \\label{{{label}}}")
        lines.append("\\end{figure}")
        return "\n".join(lines)

    if figure_plan and figure_plan.subfigures:
        latex_blocks.append(_render_subfigures())
        return "\n".join(latex_blocks)

    width = figure_plan.width if figure_plan else 1.0
    width_str = f"{width:.2f}\\linewidth" if width and width < 0.995 else "\\linewidth"
    label = _label()
    for relative in figure_paths:
        comments = _quality_comments()
        lines = comments + [
            f"\\begin{{figure}}[{_float_spec()}]",
            "  \\centering",
            f"  \\includegraphics[width={width_str}]{{{relative}}}",
            f"  \\caption{{{_caption()}}}",
        ]
        if label:
            lines.append(f"  \\label{{{label}}}")
        lines.append("\\end{figure}")
        latex_blocks.append("\n".join(lines))
    return "\n".join(latex_blocks)


def block_to_latex(
    block: common.PlanBlock,
    snippet_map: Dict[str, str],
    assets_dir: Path,
    used_assets: set[str],
    *,
    chunk_lookup: Dict[str, common.Chunk] | None = None,
    references: ReferenceIndex | None = None,
    document_class: str = "article",
    figure_agent: FigureTableAgent | None = None,
    bibliography_result: BibliographyResult | None = None,
) -> str:
    raw_snippet = snippet_map.get(block.chunk_id, "")
    cleaned = _sanitize_snippet(raw_snippet)
    snippet = cleaned if cleaned else raw_snippet
    if bibliography_result:
        snippet = bibliography_result.apply_to_text(snippet)
    if block.block_type == "figure":
        chunk = chunk_lookup.get(block.chunk_id) if chunk_lookup else None
        figure_plan = None
        if figure_agent:
            chunk_text = chunk.text if chunk else snippet
            figure_plan = figure_agent.prepare_render_plan(block, snippet, chunk_text, references)
        figures = render_figure(
            block,
            assets_dir,
            used_assets,
            references=references,
            figure_plan=figure_plan,
        ) if block.images else ""
        return figures or snippet
    figures = render_figure(block, assets_dir, used_assets, references=references) if block.images else ""
    if block.block_type == "section":
        level = (block.metadata or {}).get("hierarchy_level")
        command = _section_command(level, document_class)
        return f"{command}{{{escape(block.label)}}}\n{snippet}\n{figures}\n"
    figures = render_figure(block, assets_dir, used_assets, references=references) if block.images else ""
    if block.block_type == "equation":
        return snippet
    if figures:
        return f"{snippet}\n{figures}\n"
    return snippet


BASE_PACKAGES = [
    {"package": "graphicx", "options": None},
    {"package": "geometry", "options": "margin=1in"},
    {"package": "float", "options": None},
    {"package": "amsmath", "options": None},
    {"package": "amsthm", "options": None},
    {"package": "booktabs", "options": None},
    {"package": "siunitx", "options": None},
    {"package": "tcolorbox", "options": "most"},
    {"package": "enumitem", "options": None},
    {"package": "hyperref", "options": None},
]
DECLARATIONS = [
    "\\newtheorem{theorem}{Theorem}",
    "\\sisetup{detect-all=true, per-mode=symbol}",
    "\\newcommand{\\Question}[1]{\\section*{Question~#1}}",
    "\\newcommand{\\Transform}[2]{#1\\rightarrow #2}",
    "\\tcbset{colback=white,colframe=black!15!white,boxrule=0.4pt,arc=2pt}",
    "\\newtcolorbox{questionbox}[2][]{title={Question~#2},#1}",
    "\\newenvironment{question}[1]{\\begin{questionbox}{#1}}{\\end{questionbox}}",
    "\\newenvironment{answer}{\\begin{tcolorbox}[title={Answer},colback=green!5]}{\\end{tcolorbox}}",
    "\\setlist{leftmargin=*}",
]


def _ensure_base_packages(packages: List[Dict[str, str | None]]) -> List[Dict[str, str | None]]:
    existing = {pkg.get("package") for pkg in packages if pkg.get("package")}
    merged = packages[:]
    for base in BASE_PACKAGES:
        if base["package"] not in existing:
            merged.append(base.copy())
    return merged


def _merge_typography_config(payload: object, default: Dict[str, object]) -> Dict[str, object]:
    if not isinstance(payload, dict):
        return dict(default)
    merged = dict(default)
    for key, value in payload.items():
        merged[key] = value
    return merged


def load_preamble_config(preamble_path: Path | None) -> Dict[str, object]:
    default_typography = {"profile": "scholarly-serif", "language": "english"}
    default_class = "book"
    default_options = "11pt"
    if preamble_path and preamble_path.exists():
        try:
            data = json.loads(preamble_path.read_text(encoding="utf-8"))
            return {
                "document_class": data.get("document_class", default_class),
                "class_options": data.get("class_options", default_options),
                "packages": _ensure_base_packages(list(data.get("packages", []))),
                "typography": _merge_typography_config(data.get("typography"), default_typography),
            }
        except Exception as exc:  # pragma: no cover - best effort
            LOGGER.warning("Failed to read preamble config (%s), using default", exc)
    return {
        "document_class": default_class,
        "class_options": default_options,
        "packages": BASE_PACKAGES.copy(),
        "typography": dict(default_typography),
    }


def build_preamble(config: Dict[str, object]) -> str:
    doc_class = config.get("document_class", "article")
    class_options = config.get("class_options", "11pt")
    packages: List[Dict[str, str | None]] = list(config.get("packages", []))
    typography_payload = config.get("typography")
    typography_data = typography_payload if isinstance(typography_payload, dict) else None
    typography_engine = TypographyEngine(typography_data)
    directives = typography_engine.directives_for(doc_class)
    packages.extend(directives.packages)
    lines = [f"\\documentclass[{class_options}]{{{doc_class}}}"]
    seen = set()
    for pkg in packages:
        name = pkg.get("package")
        if not name:
            continue
        if name in seen:
            continue
        seen.add(name)
        options = pkg.get("options")
        if options:
            lines.append(f"\\usepackage[{options}]{{{name}}}")
        else:
            lines.append(f"\\usepackage{{{name}}}")
    lines.extend(directives.pre_document_commands)
    extra_cmds = config.get("extra_preamble_commands") or []
    lines.extend(extra_cmds)
    lines.extend(DECLARATIONS)
    lines.append("\\begin{document}")
    return "\n".join(lines)


def write_tex(
    plan_path: Path,
    snippets_path: Path,
    output_dir: Path,
    title: str,
    author: str,
    preamble_path: Path | None,
    sanitize_unicode: bool = True,
    chunks_path: Path | None = None,
    figure_agent: FigureTableAgent | None = None,
    style_detector: StyleDetector | None = None,
    style_profile: DocumentStyle | None = None,
    citation_detector: CitationDetector | None = None,
    bibliography_generator: BibliographyGenerator | None = None,
    citation_report: CitationReport | None = None,
    bibliography_result: BibliographyResult | None = None,
) -> Path:
    plan = common.load_plan(plan_path)
    agent = figure_agent or FigureTableAgent()
    agent.analyze_plan(plan)
    chunk_map = {}
    if chunks_path and chunks_path.exists():
        chunk_map = {chunk.chunk_id: chunk for chunk in common.load_chunks(chunks_path)}
    snippets = {snippet.chunk_id: snippet.latex for snippet in common.load_snippets(snippets_path)}
    analyze_plan(plan, chunk_map)
    snippets, reference_index = resolve_references(plan, snippets)
    assets_dir = output_dir / "assets"
    detector = style_detector or StyleDetector()
    if style_profile is None:
        style_profile = detector.detect_from_plan(plan, chunk_map)
    preamble_config = load_preamble_config(preamble_path)
    preamble_config.setdefault("style_profile", style_profile.to_dict())
    packages = preamble_config.get("packages", [])
    packages.extend(agent.required_packages())
    packages.extend(style_profile.packages)
    citation_detector = citation_detector or CitationDetector()
    if citation_report is None:
        citation_report = citation_detector.detect(plan, chunk_map)
    bibliography_generator = bibliography_generator or BibliographyGenerator()
    if bibliography_result is None:
        bibliography_result = bibliography_generator.build(citation_report, getattr(style_profile, "style_family", None))
    if bibliography_result.packages:
        packages.extend(bibliography_result.packages)
    extra_commands = preamble_config.get("extra_preamble_commands", [])
    extra_commands.extend(bibliography_result.preamble_commands)
    preamble_config["extra_preamble_commands"] = extra_commands
    preamble_config["packages"] = packages
    if style_profile.document_class:
        preamble_config["document_class"] = style_profile.document_class
    if style_profile.class_options:
        preamble_config["class_options"] = style_profile.class_options
    document_class = preamble_config.get("document_class", "article")
    preamble = build_preamble(preamble_config)
    body = []
    used_assets: set[str] = set()
    for block in plan:
        body.append(
            block_to_latex(
                block,
                snippets,
                assets_dir,
                used_assets,
                chunk_lookup=chunk_map,
                references=reference_index,
                document_class=document_class,
                figure_agent=agent,
                bibliography_result=bibliography_result,
            )
        )
    bibliography_tex = bibliography_result.latex if bibliography_result else ""
    tex_content = "\n".join(
        [
            preamble,
            f"\\title{{{escape(title)}}}",
            f"\\author{{{escape(author)}}}",
            "\\date{\\today}",
            "\\maketitle",
            *body,
            bibliography_tex,
            POSTAMBLE,
        ]
    )
    if sanitize_unicode:
        tex_content = sanitize_unicode_to_latex(tex_content)
    tex_path = output_dir / "main.tex"
    output_dir.mkdir(parents=True, exist_ok=True)
    tex_path.write_text(tex_content, encoding="utf-8")
    citations_path = output_dir / "citations.json"
    citations_payload = citation_report.to_json()
    citations_payload["bibliography"] = [entry.__dict__ for entry in (bibliography_result.entries if bibliography_result else [])]
    citations_path.write_text(json.dumps(citations_payload, indent=2), encoding="utf-8")
    LOGGER.info("LaTeX document written to %s", tex_path)
    return tex_path


@dataclass
class ProgressiveAssembler:
    """Build main.tex via structured templates with validation gates."""

    plan_path: Path
    snippets_path: Path
    preamble_path: Path | None
    output_dir: Path
    title: str
    author: str
    chunks_path: Path | None = None
    templating_engine: LaTeXTemplatingEngine = field(default_factory=LaTeXTemplatingEngine)
    symbol_normalizer: SymbolNormalizer = field(default_factory=SymbolNormalizer)
    env_detector: MathEnvironmentDetector = field(default_factory=MathEnvironmentDetector)
    consistency_validator: MathConsistencyValidator = field(default_factory=MathConsistencyValidator)
    error_repair: LaTeXErrorRepair = field(default_factory=LaTeXErrorRepair)
    domain_profile: Dict[str, object] | None = None

    def __post_init__(self) -> None:
        self.assets_dir = self.output_dir / "assets"
        self._used_assets: set[str] = set()
        self.figure_agent = FigureTableAgent()
        self.style_detector = StyleDetector()
        self.style_profile: DocumentStyle | None = None
        self.citation_detector = CitationDetector()
        self.bibliography_generator = BibliographyGenerator()
        self.citation_report: CitationReport | None = None
        self.bibliography_result: BibliographyResult | None = None

    def build(self, sanitize_unicode: bool = True) -> Path:
        preamble_config = load_preamble_config(self.preamble_path)
        plan = common.load_plan(self.plan_path)
        self.figure_agent.analyze_plan(plan)
        snippets = {snippet.chunk_id: snippet.latex for snippet in common.load_snippets(self.snippets_path)}
        chunk_map = self._load_chunks()
        analyze_plan(plan, chunk_map)
        snippets, reference_index = resolve_references(plan, snippets)
        self.style_profile = self.style_detector.detect_from_plan(plan, chunk_map)
        self.citation_report = self.citation_detector.detect(plan, chunk_map)
        self.bibliography_result = self.bibliography_generator.build(
            self.citation_report,
            getattr(self.style_profile, "style_family", None),
        )
        packages = preamble_config.get("packages", [])
        packages.extend(self.figure_agent.required_packages())
        packages.extend(self._domain_packages())
        if self.style_profile:
            packages.extend(self.style_profile.packages)
            preamble_config.setdefault("style_profile", self.style_profile.to_dict())
            if self.style_profile.document_class:
                preamble_config["document_class"] = self.style_profile.document_class
            if self.style_profile.class_options:
                preamble_config["class_options"] = self.style_profile.class_options
        if self.bibliography_result:
            packages.extend(self.bibliography_result.packages)
            extra_cmds = preamble_config.get("extra_preamble_commands", [])
            extra_cmds.extend(self.bibliography_result.preamble_commands)
            preamble_config["extra_preamble_commands"] = extra_cmds
        if self.domain_profile:
            preamble_config.setdefault("domain_profile", self.domain_profile)
        preamble_config["packages"] = packages
        document_class = preamble_config.get("document_class", "article")
        bundle = self._structure(
            plan,
            snippets,
            chunk_map,
            normalize=sanitize_unicode,
            references=reference_index,
            document_class=document_class,
            figure_agent=self.figure_agent,
            style_profile=self.style_profile,
            bibliography_result=self.bibliography_result,
        )
        if not self._should_use_template(bundle):
            LOGGER.info("Structured template skipped; using sequential assembly instead.")
            return write_tex(
                self.plan_path,
                self.snippets_path,
                self.output_dir,
                self.title,
                self.author,
                self.preamble_path,
                sanitize_unicode=sanitize_unicode,
                chunks_path=self.chunks_path,
                figure_agent=self.figure_agent,
                style_detector=self.style_detector,
                style_profile=self.style_profile,
                citation_detector=self.citation_detector,
                bibliography_generator=self.bibliography_generator,
                citation_report=self.citation_report,
                bibliography_result=self.bibliography_result,
            )
        template_name = self._select_template(bundle)
        try:
            context = self._build_context(template_name, bundle)
            tex_content = self.templating_engine.render(template_name, context)
        except Exception as exc:  # pragma: no cover - template errors are environment specific
            LOGGER.warning("Template rendering failed (%s); falling back to sequential assembly.", exc)
            return write_tex(
                self.plan_path,
                self.snippets_path,
                self.output_dir,
                self.title,
                self.author,
                self.preamble_path,
                sanitize_unicode=sanitize_unicode,
                chunks_path=self.chunks_path,
                figure_agent=self.figure_agent,
                style_detector=self.style_detector,
                style_profile=self.style_profile,
                citation_detector=self.citation_detector,
                bibliography_generator=self.bibliography_generator,
                citation_report=self.citation_report,
                bibliography_result=self.bibliography_result,
            )
        if self.bibliography_result and self.bibliography_result.latex:
            tex_content = "\n".join([tex_content, self.bibliography_result.latex])
        if sanitize_unicode:
            tex_content = sanitize_unicode_to_latex(tex_content)
        tex_path = self.output_dir / "main.tex"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        tex_path.write_text(tex_content, encoding="utf-8")
        self._write_consistency(bundle["consistency"])
        if self.citation_report:
            citations_path = self.output_dir / "citations.json"
            payload = self.citation_report.to_json()
            payload["bibliography"] = [entry.__dict__ for entry in (self.bibliography_result.entries if self.bibliography_result else [])]
            citations_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return tex_path

    def _domain_packages(self) -> List[Dict[str, str | None]]:
        profile = self.domain_profile if isinstance(self.domain_profile, dict) else {}
        specs = profile.get("recommended_packages", []) if isinstance(profile, dict) else []
        packages: List[Dict[str, str | None]] = []
        for entry in specs:
            if isinstance(entry, dict):
                name = entry.get("package")
                if name:
                    packages.append({"package": name, "options": entry.get("options")})
            elif isinstance(entry, str):
                packages.append({"package": entry, "options": None})
        return packages

    def _load_chunks(self) -> Dict[str, common.Chunk]:
        if not self.chunks_path or not self.chunks_path.exists():
            return {}
        return {chunk.chunk_id: chunk for chunk in common.load_chunks(self.chunks_path)}

    def _structure(
        self,
        plan: List[common.PlanBlock],
        snippets: Dict[str, str],
        chunk_map: Dict[str, common.Chunk],
        *,
        normalize: bool,
        references: ReferenceIndex,
        document_class: str,
        figure_agent: FigureTableAgent | None = None,
        style_profile: DocumentStyle | None = None,
        bibliography_result: BibliographyResult | None = None,
        **_unused: object,
    ) -> Dict[str, object]:
        questions: List[Dict[str, object]] = []
        proofs: List[Dict[str, object]] = []
        problems: List[Dict[str, object]] = []
        consistency: Dict[str, Dict[str, float]] = {}
        normalized_map: Dict[str, str] = {}
        fallback_body: List[str] = []
        for block in plan:
            snippet = snippets.get(block.chunk_id, "")
            normalized = self.symbol_normalizer.normalize(snippet) if normalize else snippet
            wrapped = self.env_detector.wrap(block.block_type, normalized, block.metadata)
            if bibliography_result:
                wrapped = bibliography_result.apply_to_text(wrapped)
            normalized_map[block.chunk_id] = wrapped
            fallback_body.append(
                block_to_latex(
                    block,
                    normalized_map,
                    self.assets_dir,
                    self._used_assets,
                    chunk_lookup=chunk_map,
                    references=references,
                    document_class=document_class,
                    figure_agent=figure_agent,
                    bibliography_result=bibliography_result,
                )
            )
            chunk = chunk_map.get(block.chunk_id)
            if chunk:
                stats = self.consistency_validator.validate(chunk.text, wrapped)
                consistency[block.chunk_id] = stats
            figure_path = self._first_figure(block)
            label_lower = (block.label or "").lower()
            entry = {
                "prompt": wrapped,
                "label": block.label,
                "figure": figure_path,
                "points": block.metadata.get("points") if block.metadata else None,
            }
            if block.block_type == "question" or (chunk and (chunk.metadata or {}).get("region_type") == "question"):
                entry["parts"] = chunk.metadata.get("question_parts") if chunk and chunk.metadata else []
                entry["math"] = wrapped if block.block_type == "equation" else None
                questions.append(entry)
            elif "proof" in label_lower or block.block_type == "proof":
                proofs.append(
                    {
                        "title": block.label or f"Proof {len(proofs) + 1}",
                        "statement": chunk.text if chunk else "",
                        "steps": (chunk.metadata or {}).get("proof_steps", []) if chunk else [],
                        "text": wrapped,
                    }
                )
            else:
                problems.append(
                    {
                        "prompt": wrapped,
                        "math": wrapped if block.block_type == "equation" else None,
                        "answer": (chunk.metadata or {}).get("answer") if chunk else None,
                        "points": entry.get("points"),
                    }
                )
        return {
            "questions": questions,
            "proofs": proofs,
            "problems": problems,
            "consistency": consistency,
            "fallback_body": "\n".join(fallback_body),
            "style_profile": style_profile.to_dict() if style_profile else None,
        }

    def _first_figure(self, block: common.PlanBlock) -> str | None:
        if not block.images:
            return None
        paths = _copy_block_images(block, self.assets_dir, self._used_assets)
        return paths[0] if paths else None

    def _select_template(self, bundle: Dict[str, object]) -> str:
        if bundle["questions"]:
            return "math_worksheet.tex"
        if bundle["proofs"]:
            return "proof_document.tex"
        return "problem_set.tex"

    def _should_use_template(self, bundle: Dict[str, object]) -> bool:
        questions: List[Dict[str, object]] = bundle.get("questions") or []  # type: ignore[assignment]
        proofs: List[Dict[str, object]] = bundle.get("proofs") or []  # type: ignore[assignment]
        if questions or proofs:
            return True
        problems: List[Dict[str, object]] = bundle.get("problems") or []  # type: ignore[assignment]
        if not problems:
            return False
        structured = sum(1 for problem in problems if problem.get("points") or problem.get("answer"))
        coverage = structured / len(problems)
        if coverage < 0.4:
            return False
        return True

    def _build_context(self, template: str, bundle: Dict[str, object]) -> Dict[str, object]:
        base = {
            "title": self.title,
            "author": self.author,
            "date": datetime.now().strftime("%B %d, %Y"),
        }
        if bundle.get("style_profile"):
            base["style_profile"] = bundle["style_profile"]
        fallback = bundle.get("fallback_body") or ""
        if template == "math_worksheet.tex":
            base["questions"] = bundle["questions"] or [{"prompt": fallback, "parts": [], "math": None}]
        elif template == "proof_document.tex":
            base["proofs"] = bundle["proofs"] or [
                {"title": "Auto Proof", "statement": "", "steps": [], "text": fallback}
            ]
        else:
            base["instructions"] = "Auto-generated by LaTeXify ProgressiveAssembler."
            base["problems"] = bundle["problems"] or [
                {"prompt": fallback, "math": None, "answer": None, "points": None}
            ]
        return base

    def _write_consistency(self, data: Dict[str, Dict[str, float]]) -> None:
        if not data:
            return
        path = self.output_dir / "consistency.json"
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def compile_tex(tex_path: Path, *, repair_agent: KimiK2LatexRepair | None = None) -> Path | None:
    global _LAST_COMPILATION_METRICS
    repair_agent = repair_agent or KimiK2LatexRepair()
    enable_robust = os.environ.get("LATEXIFY_ENABLE_ROBUST_COMPILATION", "1").lower() not in {"0", "false", "off"}
    retry_count = max(1, int(os.environ.get("LATEXIFY_COMPILATION_RETRY_COUNT", "3")))
    if enable_robust:
        repair_agent.preflight_check(tex_path)
        result = run_robust_compilation(
            tex_path,
            cache_dir=tex_path.parent / ".latexify_cache",
            repair_agent=repair_agent,
            max_retries=retry_count,
        )
        _LAST_COMPILATION_METRICS = {
            "robust_enabled": True,
            "compilation_attempts": len(result.attempt_history),
            "attempt_history": result.attempt_history,
            "section_reports": result.section_reports,
            "error_summary": result.error_summary,
            "recovery_success": result.recovery_success,
        }
        return result.pdf_path if result.success else None
    pdf_path, attempts = _traditional_compile(tex_path)
    _LAST_COMPILATION_METRICS = {
        "robust_enabled": False,
        "compilation_attempts": len(attempts),
        "attempt_history": attempts,
        "section_reports": [],
        "error_summary": [],
        "recovery_success": False,
    }
    return pdf_path


def _traditional_compile(tex_path: Path) -> tuple[Path | None, List[Dict[str, object]]]:
    pdf_path = tex_path.with_suffix(".pdf")
    attempts: List[Dict[str, object]] = []
    engines = ["tectonic", "latexmk"]
    for engine in engines:
        binary = shutil.which(engine)
        if not binary:
            continue
        if engine == "tectonic":
            cmd = [binary, "--keep-logs", "--keep-intermediates", str(tex_path)]
        else:
            cmd = [binary, "-pdf", "-silent", str(tex_path)]
        try:
            subprocess.run(cmd, check=True, cwd=str(tex_path.parent))
            if pdf_path.exists():
                attempts.append({"engine": engine, "success": True})
                return pdf_path, attempts
        except subprocess.CalledProcessError as exc:  # pragma: no cover - depends on host
            LOGGER.warning("TeX compilation failed with %s: %s", engine, exc)
            attempts.append({"engine": engine, "success": False})
    LOGGER.warning("No TeX compiler available; skipping PDF generation")
    return None, attempts


def consume_compilation_metrics() -> Dict[str, object] | None:
    global _LAST_COMPILATION_METRICS
    metrics = _LAST_COMPILATION_METRICS
    _LAST_COMPILATION_METRICS = None
    return metrics


def run_assembly(
    plan_path: Path,
    snippets_path: Path,
    preamble_path: Path | None,
    output_dir: Path,
    title: str,
    author: str,
    skip_compile: bool = False,
    use_unicode_sanitizer: bool = True,
    chunks_path: Path | None = None,
    refinement_passes: int = 3,
    domain_profile: Dict[str, object] | None = None,
) -> Path:
    assembler = ProgressiveAssembler(
        plan_path=plan_path,
        snippets_path=snippets_path,
        preamble_path=preamble_path,
        output_dir=output_dir,
        title=title,
        author=author,
        chunks_path=chunks_path,
        domain_profile=domain_profile,
    )
    tex_path = assembler.build(sanitize_unicode=use_unicode_sanitizer)
    repair_agent = KimiK2LatexRepair()
    compile_wrapper = lambda target: compile_tex(target, repair_agent=repair_agent)
    refinement_report: Dict[str, object] | None = None
    refinement_agent = RefinementAgent(compile_callable=compile_wrapper)
    if not skip_compile:
        repair_agent.preflight_check(tex_path)
        refinement_report = refinement_agent.refine(tex_path, max_passes=refinement_passes)
        refinement_report.setdefault("history", [])
        if not refinement_report.get("success"):
            log_path = tex_path.with_suffix(".log")
            if assembler.error_repair.repair(tex_path, log_path):
                secondary = refinement_agent.refine(tex_path, max_passes=max(1, refinement_passes - 1))
                refinement_report["passes"].extend(secondary.get("passes", []))
                refinement_report.setdefault("history", []).extend(secondary.get("history", []))
                refinement_report["success"] = secondary.get("success", False)
        report_path = tex_path.parent / "refinement_report.json"
        report_path.write_text(json.dumps(refinement_report or {}, indent=2), encoding="utf-8")
    return tex_path


__all__ = ["run_assembly", "ProgressiveAssembler", "compile_tex", "consume_compilation_metrics"]
