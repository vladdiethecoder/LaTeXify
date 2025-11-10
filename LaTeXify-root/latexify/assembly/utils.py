from __future__ import annotations

import re
from typing import Iterable, List, Sequence

_ESCAPE_MAP = str.maketrans({
    "\\": r"\\textbackslash{}",
    "{": r"\{",
    "}": r"\}",
    "%": r"\%",
    "#": r"\#",
    "&": r"\&",
    "$": r"\$",
    "_": r"\_",
})


def escape_tex(value: str) -> str:
    return value.translate(_ESCAPE_MAP)


def text_to_paragraphs(text: str) -> str:
    chunks = [segment.strip() for segment in re.split(r"\n\s*\n", text or "") if segment.strip()]
    if not chunks:
        return ""
    return "\n\n".join(escape_tex(chunk) for chunk in chunks) + "\n"


def parse_table_rows(text: str) -> List[List[str]]:
    rows: List[List[str]] = []
    for line in (text or "").splitlines():
        raw = line.strip().strip("|")
        if not raw:
            continue
        if "|" in raw:
            cells = [cell.strip() for cell in raw.split("|")]
        else:
            cells = [cell.strip() for cell in re.split(r"\s{2,}", raw) if cell.strip()]
        if cells:
            rows.append([escape_tex(c) for c in cells])
    return rows


def _unique(values: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        if not value:
            continue
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def inject_packages(preamble: str, packages: Sequence[str]) -> str:
    pkg_lines = _unique(pkg.strip() for pkg in packages)
    lines = preamble.splitlines()
    if not pkg_lines:
        return preamble if preamble.endswith("\n") else preamble + "\n"
    insertion_idx = 1 if lines else 0
    new_lines = lines[:insertion_idx] + pkg_lines + lines[insertion_idx:]
    result = "\n".join(new_lines)
    return result if result.endswith("\n") else result + "\n"


__all__ = [
    "escape_tex",
    "inject_packages",
    "parse_table_rows",
    "text_to_paragraphs",
    "_unique",
]
