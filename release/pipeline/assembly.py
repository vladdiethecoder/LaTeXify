"""Assembly stage that creates the final LaTeX document and optional PDF."""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from ..core import common
from ..core.sanitizer import sanitize_unicode_to_latex
from .consistency import MathConsistencyValidator
from .error_repair import LaTeXErrorRepair
from .math_environment import MathEnvironmentDetector
from .symbol_normalizer import SymbolNormalizer
from .template_engine import LaTeXTemplatingEngine

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
    return "\n".join(cleaned_lines).strip()


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


def render_figure(block: common.PlanBlock, assets_dir: Path, used_assets: set[str]) -> str:
    latex_blocks: List[str] = []
    figure_paths = _copy_block_images(block, assets_dir, used_assets)
    for relative in figure_paths:
        latex_blocks.append(
            "\n".join(
                [
                    "\\begin{figure}[H]",
                    "  \\centering",
                    f"  \\includegraphics[width=\\linewidth]{{{relative}}}",
                    f"  \\caption{{{escape(block.label)}}}",
                    "\\end{figure}",
                ]
            )
        )
    return "\n".join(latex_blocks)


def block_to_latex(block: common.PlanBlock, snippet_map: Dict[str, str], assets_dir: Path, used_assets: set[str]) -> str:
    raw_snippet = snippet_map.get(block.chunk_id, "")
    cleaned = _sanitize_snippet(raw_snippet)
    snippet = cleaned if cleaned else raw_snippet
    figures = render_figure(block, assets_dir, used_assets) if block.images else ""
    if block.block_type == "section":
        return f"\\section{{{escape(block.label)}}}\n{snippet}\n{figures}\n"
    if block.block_type == "figure":
        return figures or snippet
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


def load_preamble_config(preamble_path: Path | None) -> Dict[str, object]:
    default = {
        "document_class": "book",
        "class_options": "11pt",
        "packages": BASE_PACKAGES.copy(),
    }
    if preamble_path and preamble_path.exists():
        try:
            data = json.loads(preamble_path.read_text(encoding="utf-8"))
            return {
                "document_class": data.get("document_class", default["document_class"]),
                "class_options": data.get("class_options", default["class_options"]),
                "packages": _ensure_base_packages(list(data.get("packages", []))),
            }
        except Exception as exc:  # pragma: no cover - best effort
            LOGGER.warning("Failed to read preamble config (%s), using default", exc)
    return default


def build_preamble(config: Dict[str, object]) -> str:
    doc_class = config.get("document_class", "article")
    class_options = config.get("class_options", "11pt")
    packages: List[Dict[str, str | None]] = list(config.get("packages", []))
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
) -> Path:
    plan = common.load_plan(plan_path)
    snippets = {snippet.chunk_id: snippet.latex for snippet in common.load_snippets(snippets_path)}
    assets_dir = output_dir / "assets"
    preamble_config = load_preamble_config(preamble_path)
    preamble = build_preamble(preamble_config)
    body = []
    used_assets: set[str] = set()
    for block in plan:
        body.append(block_to_latex(block, snippets, assets_dir, used_assets))
    tex_content = "\n".join(
        [
            preamble,
            f"\\title{{{escape(title)}}}",
            f"\\author{{{escape(author)}}}",
            "\\date{\\today}",
            "\\maketitle",
            *body,
            POSTAMBLE,
        ]
    )
    if sanitize_unicode:
        tex_content = sanitize_unicode_to_latex(tex_content)
    tex_path = output_dir / "main.tex"
    output_dir.mkdir(parents=True, exist_ok=True)
    tex_path.write_text(tex_content, encoding="utf-8")
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

    def __post_init__(self) -> None:
        self.assets_dir = self.output_dir / "assets"
        self._used_assets: set[str] = set()

    def build(self, sanitize_unicode: bool = True) -> Path:
        plan = common.load_plan(self.plan_path)
        snippets = {snippet.chunk_id: snippet.latex for snippet in common.load_snippets(self.snippets_path)}
        chunk_map = self._load_chunks()
        bundle = self._structure(plan, snippets, chunk_map, normalize=sanitize_unicode)
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
            )
        if sanitize_unicode:
            tex_content = sanitize_unicode_to_latex(tex_content)
        tex_path = self.output_dir / "main.tex"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        tex_path.write_text(tex_content, encoding="utf-8")
        self._write_consistency(bundle["consistency"])
        return tex_path

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
            normalized_map[block.chunk_id] = wrapped
            fallback_body.append(block_to_latex(block, normalized_map, self.assets_dir, self._used_assets))
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

    def _build_context(self, template: str, bundle: Dict[str, object]) -> Dict[str, object]:
        base = {
            "title": self.title,
            "author": self.author,
            "date": datetime.now().strftime("%B %d, %Y"),
        }
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


def compile_tex(tex_path: Path) -> Path | None:
    pdf_path = tex_path.with_suffix(".pdf")
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
                LOGGER.info("PDF generated via %s", engine)
                return pdf_path
        except subprocess.CalledProcessError as exc:  # pragma: no cover - depends on host
            LOGGER.warning("TeX compilation failed with %s: %s", engine, exc)
    LOGGER.warning("No TeX compiler available; skipping PDF generation")
    return None


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
) -> Path:
    assembler = ProgressiveAssembler(
        plan_path=plan_path,
        snippets_path=snippets_path,
        preamble_path=preamble_path,
        output_dir=output_dir,
        title=title,
        author=author,
        chunks_path=chunks_path,
    )
    tex_path = assembler.build(sanitize_unicode=use_unicode_sanitizer)
    if not skip_compile:
        pdf = compile_tex(tex_path)
        if pdf is None:
            log_path = tex_path.with_suffix(".log")
            if assembler.error_repair.repair(tex_path, log_path):
                compile_tex(tex_path)
    return tex_path


__all__ = ["run_assembly", "ProgressiveAssembler"]
