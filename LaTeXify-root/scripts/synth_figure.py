"""Specialized figure synthesis helper."""

from __future__ import annotations

from typing import Dict, List, Tuple


def synthesize(bundle: Dict) -> Tuple[str, List[str]]:
    image_path = bundle.get("image_path") or "figure-placeholder.pdf"
    caption = bundle.get("caption") or (bundle.get("prompt") or "Auto-generated figure").splitlines()[0]
    label = bundle.get("id") or "fig"
    lines = [
        "\\begin{figure}[ht]",
        "  \\centering",
        f"  \\includegraphics[width=\\linewidth]{{{image_path}}}",
        f"  \\caption{{{caption}}}",
        f"  \\label{{fig:{label}}}",
        "\\end{figure}",
    ]
    return "\n".join(lines) + "\n", ["graphicx"]
