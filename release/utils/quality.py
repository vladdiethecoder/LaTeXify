"""Simple LaTeX quality checks used after assembly."""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import List

TEMPLATE_TOKENS = [
    "Style exemplar",
    "Baseline snippet",
    "Respond with",
    "<latex>",
    "<<<SOURCE",
    "SOURCE",
    "Auto-captioned figure",
    "TODO",
]


def inspect_tex(tex_path: Path) -> List[str]:
    issues: List[str] = []
    latex = tex_path.read_text(encoding="utf-8")
    for token in TEMPLATE_TOKENS:
        if token in latex:
            issues.append(f"Template token '{token}' detected in {tex_path.name}.")
    figures = re.findall(r"\\includegraphics\[.*?\]{([^}]+)}", latex)
    if figures:
        dupes = [name for name, count in Counter(figures).items() if count > 1]
        if dupes:
            issues.append(f"Duplicate figure assets detected: {', '.join(sorted(dupes))}")
        env_count = latex.count("\\begin{figure}")
        if env_count != len(figures):
            issues.append(
                f"Mismatch between figure environments ({env_count}) and includegraphics calls ({len(figures)})."
            )
    return issues


__all__ = ["inspect_tex"]
