from __future__ import annotations

"""Shared helpers for deterministic LaTeX synthesis."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import json
import re


_ESCAPE_RE = re.compile(r"[\\{}_\%$]")


def sanitize_inline(text: str | None) -> str:
    """Escape characters that would break LaTeX when used inline."""

    def _replace(match: re.Match[str]) -> str:
        ch = match.group(0)
        if ch == "\\":
            return r"\\textbackslash{}"
        return "\\" + ch

    return _ESCAPE_RE.sub(_replace, text or "")


def title_from_question(question: str, default: str) -> str:
    if ":" in question:
        return question.split(":", 1)[1].strip() or default
    return question.strip() or default


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return slug or "section"


def collect_text_segments(values: Sequence | None) -> List[str]:
    texts: List[str] = []
    if not values:
        return texts
    for entry in values:
        if isinstance(entry, dict):
            txt = entry.get("text") or entry.get("content") or ""
        else:
            txt = str(entry)
        if txt:
            texts.append(str(txt))
    return texts


def user_flag_uncertain(bundle: Dict) -> bool:
    flags = bundle.get("user_answer", {}).get("flags", {})
    if isinstance(flags, dict):
        return bool(flags.get("ocr_uncertain"))
    return False


CAPABILITY_HINTS = [
    ("amsmath", ["\\begin{align", "\\begin{equation", "\\[", "\\("] ),
    ("amssymb", ["\\mathbb{", "\\mathcal{", "\\mathfrak{"] ),
    ("graphicx", ["\\includegraphics"]),
    ("booktabs", ["\\toprule", "\\midrule", "\\bottomrule"]),
    ("hyperref", ["\\url{", "\\href{"] ),
]


def capabilities_from_text(tex: str, hints: Iterable[tuple[str, Iterable[str]]] | None = None) -> List[str]:
    mapping = hints or CAPABILITY_HINTS
    caps: List[str] = []
    for name, needles in mapping:
        if any(n in tex for n in needles):
            caps.append(name)
    # deterministic order
    seen = set()
    ordered: List[str] = []
    for cap in caps:
        if cap not in seen:
            seen.add(cap)
            ordered.append(cap)
    return ordered


@dataclass(frozen=True)
class SpecialistPrompt:
    version: str
    body: str


_SPECIALIST_PROMPT_CACHE: SpecialistPrompt | None = None


def _extract_prompt_version(text: str) -> str:
    match = re.search(r'"prompt_version"\s*:\s*"([^"]+)"', text)
    if match:
        return match.group(1)
    # Fallback to simple heuristic based on heading
    heading = next((line.strip("# ") for line in text.splitlines() if line.startswith("#")), "")
    return heading or "unknown"


def load_specialist_prompt(root: Path | None = None) -> SpecialistPrompt:
    """Load the specialist prompt definition from tasks/synthesis_agent.md."""
    global _SPECIALIST_PROMPT_CACHE
    if _SPECIALIST_PROMPT_CACHE is not None:
        return _SPECIALIST_PROMPT_CACHE
    if root is None:
        root = Path(__file__).resolve().parents[2]
    prompt_path = root / "tasks" / "synthesis_agent.md"
    text = prompt_path.read_text(encoding="utf-8")
    version = _extract_prompt_version(text)
    prompt = SpecialistPrompt(version=version, body=text)
    _SPECIALIST_PROMPT_CACHE = prompt
    return prompt


def read_bundle(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))
