"""Automatically insert explanatory comments before section headings."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, List

from ..models.vllm_client import get_vllm_client

LOGGER = logging.getLogger(__name__)
SECTION_RE = re.compile(r"^(?P<indent>\s*)\\section\*?\{(?P<title>[^}]*)\}", re.MULTILINE)
PROMPT = """You provide brief helpful comments for LaTeX sections.
Given the list of section titles, return JSON mapping each title to a short (<80 char) comment.

Sections:
{sections}

JSON:
"""


def add_section_comments(tex_path: Path) -> Dict[str, object]:
    text = tex_path.read_text(encoding="utf-8")
    titles: List[str] = []
    for match in SECTION_RE.finditer(text):
        title = match.group("title").strip()
        if title and title not in titles:
            titles.append(title)
    report = {"added": 0, "sections": len(titles)}
    if not titles:
        return report
    client = get_vllm_client()
    comments: Dict[str, str] = {}
    if client is not None:
        prompt = PROMPT.format(sections="\n".join(f"- {title}" for title in titles))
        try:  # pragma: no cover - depends on vLLM runtime
            response = client.generate(prompt, stop=[], max_tokens=256).strip()
            if response.lower().startswith("<latex>"):
                response = response.split("</latex>", 1)[0]
            data = json.loads(response)
            if isinstance(data, dict):
                for title, comment in data.items():
                    if isinstance(comment, str):
                        comments[title.strip()] = comment.strip()
        except Exception as exc:
            LOGGER.debug("Comment generation LLM failed: %s", exc)
    if not comments:
        return report
    lines = text.splitlines()
    new_lines: List[str] = []
    added = 0
    for line in lines:
        match = re.match(r"(\s*\\section\*?\{([^}]*)\})", line)
        if match:
            title = match.group(2).strip()
            comment = comments.get(title)
            if comment:
                indent = re.match(r"\s*", line).group(0)
                new_lines.append(f"{indent}% {comment}")
                added += 1
        new_lines.append(line)
    tex_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    report["added"] = added
    return report


__all__ = ["add_section_comments"]
