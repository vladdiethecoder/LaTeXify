"""Template-driven LaTeX generation utilities."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable

from jinja2 import Environment, FileSystemLoader, select_autoescape

TEMPLATE_DIR = Path(__file__).resolve().parents[1] / "templates"


@dataclass
class LaTeXTemplatingEngine:
    template_dir: Path = TEMPLATE_DIR

    def __post_init__(self) -> None:
        self.env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=select_autoescape(enabled_extensions=("tex",)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def available_templates(self) -> Iterable[str]:
        return sorted(name for name in self.env.list_templates(extensions=["tex"]))

    def render(self, template_name: str, context: Dict[str, Any]) -> str:
        template = self.env.get_template(template_name)
        return template.render(**context)

    def render_math_worksheet(self, payload: Dict[str, Any]) -> str:
        return self.render("math_worksheet.tex", payload)

    def render_proof_document(self, payload: Dict[str, Any]) -> str:
        return self.render("proof_document.tex", payload)

    def render_problem_set(self, payload: Dict[str, Any]) -> str:
        return self.render("problem_set.tex", payload)


def save_rendered(tex: str, path: Path) -> Path:
    path.write_text(tex, encoding="utf-8")
    return path


__all__ = ["LaTeXTemplatingEngine", "save_rendered"]
