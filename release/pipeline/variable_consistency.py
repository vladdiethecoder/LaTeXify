"""LLM-assisted math variable consistency normalization."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from ..models.vllm_client import get_vllm_client

LOGGER = logging.getLogger(__name__)
INLINE_MATH_RE = re.compile(r"\$(?:\\.|[^$])+\$")
DISPLAY_MATH_RE = re.compile(r"\\\[([\s\S]*?)\\]", re.MULTILINE)
PAREN_MATH_RE = re.compile(r"\\\(([\s\S]*?)\\\)", re.MULTILINE)
ENV_MATH_RE = re.compile(
    r"\\begin\{(equation\*?|align\*?|gather\*?|multline\*?)\}([\s\S]*?)\\end\{\1\}",
    re.MULTILINE,
)
TOKEN_RE = re.compile(r"(?<![\\A-Za-z])([A-Za-z])(?![A-Za-z])")
PROMPT = """You ensure consistent variable case in math expressions. \
Each conflict lists the base letter and the forms found. Choose the preferred form \
and respond with JSON mapping base letter -> preferred character.

Conflicts:
{conflicts}

JSON:
"""


@dataclass
class Conflict:
    base: str
    forms: List[str]
    samples: List[str]


def _math_spans(text: str) -> List[Tuple[int, int, str]]:
    spans: List[Tuple[int, int, str]] = []
    for regex in (ENV_MATH_RE, DISPLAY_MATH_RE, PAREN_MATH_RE, INLINE_MATH_RE):
        for match in regex.finditer(text):
            start, end = match.span()
            content = match.group(0)
            spans.append((start, end, content))
    spans.sort(key=lambda item: item[0])
    return spans


def _collect_conflicts(spans: Sequence[Tuple[int, int, str]]) -> Dict[str, Conflict]:
    conflicts: Dict[str, Conflict] = {}
    for _, _, content in spans:
        for token_match in TOKEN_RE.finditer(content):
            token = token_match.group(1)
            base = token.lower()
            record = conflicts.setdefault(base, Conflict(base=base, forms=[], samples=[]))
            if token not in record.forms:
                record.forms.append(token)
                if len(record.samples) < 3:
                    snippet = content.strip()
                    record.samples.append(snippet[:200])
    return {base: conflict for base, conflict in conflicts.items() if len(conflict.forms) > 1}


def _build_prompt(conflicts: Dict[str, Conflict]) -> str:
    lines: List[str] = []
    for idx, conflict in enumerate(conflicts.values(), start=1):
        lines.append(
            f"{idx}. base '{conflict.base}' forms: {', '.join(conflict.forms)}. "
            f"Sample: {conflict.samples[0]}"
        )
    return PROMPT.format(conflicts="\n".join(lines))


def _apply_replacements(text: str, spans: List[Tuple[int, int, str]], mapping: Dict[str, str]) -> str:
    if not mapping:
        return text
    parts: List[str] = []
    cursor = 0
    for start, end, content in spans:
        parts.append(text[cursor:start])
        updated = content
        for base, target in mapping.items():
            target = target.strip()
            if not target or len(target) != 1:
                continue
            target_lower = target.lower()
            for form in set(TOKEN_RE.findall(content)):
                if form.lower() == base and form != target:
                    pattern = re.compile(rf"(?<![\\A-Za-z]){re.escape(form)}(?![A-Za-z])")
                    updated = pattern.sub(target, updated)
        parts.append(updated)
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts)


def normalize_variables(tex_path: Path) -> Dict[str, object]:
    tex = tex_path.read_text(encoding="utf-8")
    spans = _math_spans(tex)
    conflicts = _collect_conflicts(spans)
    report = {"changed": False, "conflicts": len(conflicts)}
    if not conflicts:
        return report
    client = get_vllm_client()
    mapping: Dict[str, str] = {}
    if client is not None:
        prompt = _build_prompt(conflicts)
        try:  # pragma: no cover - depends on vLLM runtime
            response = client.generate(prompt, stop=[], max_tokens=256).strip()
            if response.lower().startswith("<latex>"):
                response = response.split("</latex>", 1)[0]
            data = json.loads(response)
            if isinstance(data, dict):
                for base, target in data.items():
                    if isinstance(base, str) and isinstance(target, str):
                        mapping[base.lower()] = target.strip()
        except Exception as exc:
            LOGGER.debug("Variable consistency LLM failed: %s", exc)
    if not mapping:
        return report
    updated = _apply_replacements(tex, spans, mapping)
    if updated != tex:
        tex_path.write_text(updated, encoding="utf-8")
        report["changed"] = True
    return report


__all__ = ["normalize_variables"]
