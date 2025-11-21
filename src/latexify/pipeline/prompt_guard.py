"""Heuristics to strip prompt-injection attempts from OCR text."""
from __future__ import annotations

import re
from typing import Iterable

PATTERNS: Iterable[re.Pattern[str]] = [
    re.compile(r"(?i)you are an?"),
    re.compile(r"(?i)ignore previous instructions"),
    re.compile(r"(?i)rewrite this"),
    re.compile(r"(?i)respond with"),
    re.compile(r"(?i)do not translate"),
    re.compile(r"(?i)as an ai language model"),
    re.compile(r"(?i)<latex>"),
    re.compile(r"(?i)meta[-\s]?instruction"),
]


def sanitize_chunk_text(text: str) -> str:
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            lines.append(line)
            continue
        if any(pattern.search(stripped) for pattern in PATTERNS):
            continue
        lines.append(line)
    sanitized = "\n".join(lines).strip()
    return sanitized or text.strip()


__all__ = ["sanitize_chunk_text"]
