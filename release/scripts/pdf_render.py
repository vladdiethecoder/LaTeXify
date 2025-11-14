"""Render PPO prompts into PDFs via structured LaTeX templates."""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List

TEMPLATES_ROOT = Path(__file__).resolve().parents[1] / "templates"
PPO_TEMPLATE_DIR = TEMPLATES_ROOT / "ppo"
TEMPLATE_REGISTRY: Dict[str, Path] = {
    "article": PPO_TEMPLATE_DIR / "article.tex",
    "notebook": PPO_TEMPLATE_DIR / "notebook.tex",
    "report": PPO_TEMPLATE_DIR / "report.tex",
    "legacy": TEMPLATES_ROOT / "ppo_prompt.tex",
}
DEFAULT_TEMPLATE = "article"


def available_templates() -> List[str]:
    """Return the list of template keys that resolve to existing files."""

    names = [name for name, path in TEMPLATE_REGISTRY.items() if path.exists()]
    if DEFAULT_TEMPLATE not in names:
        names.append(DEFAULT_TEMPLATE)
    return sorted(set(names))


def _escape_latex(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "{": r"\{",
        "}": r"\}",
        "#": r"\#",
        "$": r"\$",
        "%": r"\%",
        "&": r"\&",
        "_": r"\_",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    pattern = re.compile("|".join(re.escape(key) for key in replacements))
    return pattern.sub(lambda match: replacements[match.group(0)], text)


def _choose_engine() -> list[str]:
    if shutil.which("tectonic"):
        return ["tectonic", "--outdir", "{outdir}", "{tex}"]
    if shutil.which("latexmk"):
        return [
            "latexmk",
            "-pdf",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-output-directory={outdir}",
            "{tex}",
        ]
    raise RuntimeError("Neither tectonic nor latexmk is available for rendering prompts")


def _load_template(name: str) -> str:
    path = TEMPLATE_REGISTRY.get(name)
    if path is None or not path.exists():
        fallback = TEMPLATE_REGISTRY.get(DEFAULT_TEMPLATE)
        if fallback and fallback.exists():
            return fallback.read_text(encoding="utf-8")
        raise FileNotFoundError(
            f"Template '{name}' not found and default template missing at {fallback}"
        )
    return path.read_text(encoding="utf-8")


def _render_markdown_table(lines: Iterable[str]) -> str:
    rows = [_parse_table_row(line) for line in lines if line.strip()]
    rows = [row for row in rows if row]
    if not rows:
        return ""
    header = rows[0]
    body = rows[1:]
    if body and _is_separator_row(body[0]):
        body = body[1:]
    columns = len(header)
    col_spec = "l" * max(columns, 1)
    lines_out = [
        "\\begin{table}[H]",
        "  \\centering",
        f"  \\begin{{tabular}}{{{col_spec}}}",
        "    \\toprule",
        f"    {' & '.join(header)} \\\\",
    ]
    if body:
        lines_out.append("    \\midrule")
        for row in body:
            padded = row + [""] * (columns - len(row))
            lines_out.append(f"    {' & '.join(padded)} \\\\")
    lines_out.extend(
        [
            "    \\bottomrule",
            "  \\end{tabular}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines_out)


def _parse_table_row(line: str) -> List[str]:
    parts = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return [_escape_latex(cell or r"\,") for cell in parts]


def _is_separator_row(cells: List[str]) -> bool:
    return all(set(cell).issubset({"-", ":", " "}) for cell in cells)


def _format_paragraph(text: str) -> str:
    cleaned = _escape_latex(" ".join(text.split()))
    if not cleaned:
        return ""
    return f"\\noindent {cleaned}\\\\"


def _flush_bullets(buffer: List[str], blocks: List[str]) -> None:
    if not buffer:
        return
    blocks.append("\\begin{itemize}")
    for item in buffer:
        blocks.append(f"  \\item {item}")
    blocks.append("\\end{itemize}")
    buffer.clear()


def format_prompt_to_latex(prompt_text: str) -> str:
    """Convert lightweight Markdown into LaTeX sections, lists, and tables."""

    lines = prompt_text.strip().splitlines()
    blocks: List[str] = []
    bullets: List[str] = []
    idx = 0
    while idx < len(lines):
        raw = lines[idx].rstrip()
        if not raw:
            _flush_bullets(bullets, blocks)
            blocks.append("")
            idx += 1
            continue
        if raw.startswith("### "):
            _flush_bullets(bullets, blocks)
            title = _escape_latex(raw[4:].strip())
            blocks.append(f"\\paragraph*{{{title}}}")
            idx += 1
            continue
        if raw.startswith("## "):
            _flush_bullets(bullets, blocks)
            title = _escape_latex(raw[3:].strip())
            blocks.append(f"\\subsection*{{{title}}}")
            idx += 1
            continue
        if raw.startswith("# "):
            _flush_bullets(bullets, blocks)
            title = _escape_latex(raw[2:].strip())
            blocks.append(f"\\section*{{{title}}}")
            idx += 1
            continue
        stripped = raw.strip()
        if stripped.startswith("- "):
            bullets.append(_escape_latex(stripped[2:].strip()))
            idx += 1
            continue
        if stripped.startswith("|") and stripped.count("|") >= 2:
            _flush_bullets(bullets, blocks)
            table_lines = []
            while idx < len(lines):
                candidate = lines[idx].strip()
                if not candidate or not candidate.startswith("|"):
                    break
                table_lines.append(candidate)
                idx += 1
            blocks.append(_render_markdown_table(table_lines))
            continue
        _flush_bullets(bullets, blocks)
        blocks.append(_format_paragraph(raw))
        idx += 1
    _flush_bullets(bullets, blocks)
    latex = "\n".join(blocks).strip()
    return latex or "\\paragraph*{(empty prompt)}"


def render_prompt_to_pdf(
    prompt_text: str,
    out_dir: Path,
    *,
    template: str = DEFAULT_TEMPLATE,
    metadata: dict[str, str] | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    template_text = _load_template(template)
    metadata = metadata or {}
    title = _escape_latex(metadata.get("title") or "Prompt Context")
    subtitle = _escape_latex(metadata.get("subtitle") or "")
    content = format_prompt_to_latex(prompt_text)
    latex = (
        template_text.replace("%%PROMPT_TITLE%%", title)
        .replace("%%PROMPT_SUBTITLE%%", subtitle)
        .replace("%%PROMPT_CONTENT%%", content)
    )
    tex_path = out_dir / "main.tex"
    tex_path.write_text(latex, encoding="utf-8")
    engine_cmd = _choose_engine()
    resolved_cmd = [arg.format(outdir=str(out_dir), tex=str(tex_path)) for arg in engine_cmd]
    subprocess.run(resolved_cmd, check=True, cwd=out_dir)
    pdf_path = out_dir / "main.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(f"Failed to produce PDF at {pdf_path}")
    return pdf_path


__all__ = ["available_templates", "format_prompt_to_latex", "render_prompt_to_pdf"]
