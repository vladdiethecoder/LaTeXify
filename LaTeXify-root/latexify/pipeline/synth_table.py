"""Specialized table synthesis helper."""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

from .synth_shared import sanitize_inline, slugify


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "").strip()


def _parse_table(html: str) -> List[List[str]]:
    rows: List[List[str]] = []
    for row_html in re.findall(r"<tr[^>]*>(.*?)</tr>", html, flags=re.S | re.I):
        cells = re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row_html, flags=re.S | re.I)
        if not cells:
            continue
        rows.append([_strip_html(cell) for cell in cells])
    return rows


def _sanitized_row(row: List[str], width: int) -> str:
    padded = row + ["--"] * (width - len(row))
    joined = " & ".join(sanitize_inline(cell or "--") for cell in padded)
    return "    " + joined + " \\"


def synthesize(bundle: Dict) -> Tuple[str, List[str]]:
    html = bundle.get("table_html") or bundle.get("html") or bundle.get("prompt") or ""
    rows = _parse_table(html)
    if not rows:
        # Fallback: treat prompt lines as simple rows
        lines = [line for line in (bundle.get("prompt") or "").splitlines() if line.strip()]
        rows = [[col.strip() for col in line.split("|")] for line in lines]
        rows = [row for row in rows if any(cell for cell in row)]
    if not rows:
        body = "% TODO: table content unavailable"
        return body + "\n", ["booktabs"]

    col_count = max(len(row) for row in rows)
    alignment = "l " * col_count
    lines = [
        "\\begin{table}[ht]",
        "  \\centering",
        f"  \\begin{{tabular}}{{{alignment.strip()}}}",
        "    \\toprule",
    ]
    header, *rest = rows
    lines.append(_sanitized_row(header, col_count))
    if rest:
        lines.append("    \\midrule")
        for row in rest:
            lines.append(_sanitized_row(row, col_count))
    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("  \\caption{Auto-generated table}")
    label = slugify(bundle.get("id") or "table")
    lines.append(f"  \\label{{tab:{label}}}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n", ["booktabs"]
