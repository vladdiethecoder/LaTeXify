"""Use an LLM to determine if preamble packages can be removed."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import List

from ..models.vllm_client import get_vllm_client

LOGGER = logging.getLogger(__name__)
PACKAGE_RE = re.compile(r"(\\usepackage(?:\[[^\]]*\])?\{([^}]+)\})")
PROMPT = """You are optimizing a LaTeX preamble. Given the package list and document excerpt,
return a JSON array of package names that can be safely removed because they are unused.
Return [] if unsure. NEVER remove packages required for math (amsmath, amssymb).

Packages:
{packages}

Excerpt:
{excerpt}

JSON array of removable package names:
"""


def optimize_preamble(tex_path: Path) -> bool:
    client = get_vllm_client()
    if client is None:
        return False
    text = tex_path.read_text(encoding="utf-8")
    matches = list(PACKAGE_RE.finditer(text))
    if not matches:
        return False
    package_lines: List[str] = []
    package_names: List[str] = []
    for idx, match in enumerate(matches, start=1):
        package_line = match.group(1)
        raw_names = match.group(2)
        for name in raw_names.split(","):
            pkg = name.strip()
            if not pkg:
                continue
            package_lines.append(f"{idx}. {pkg}: {package_line}")
            package_names.append(pkg)
    excerpt = "\n".join(text.splitlines()[:200])
    prompt = PROMPT.format(packages="\n".join(package_lines), excerpt=excerpt)
    try:  # pragma: no cover - depends on vLLM runtime
        response = client.generate(prompt, stop=[], max_tokens=256)
    except Exception as exc:
        LOGGER.debug("Preamble optimizer LLM failed: %s", exc)
        return False
    cleaned = response.strip()
    if cleaned.lower().startswith("<latex>"):
        cleaned = cleaned.split("</latex>", 1)[0]
    try:
        removable = json.loads(cleaned)
    except Exception:
        LOGGER.debug("Preamble optimizer returned non-JSON payload: %s", cleaned)
        return False
    if not isinstance(removable, list):
        return False
    removable_set = {
        name.strip() for name in removable if isinstance(name, str) and name.strip() in package_names
    }
    if not removable_set:
        return False
    lines = text.splitlines()
    updated_lines = []
    for line in lines:
        match = PACKAGE_RE.search(line)
        if match:
            pkg_group = match.group(2)
            if any(name.strip() in removable_set for name in pkg_group.split(",")):
                LOGGER.info("Preamble optimizer removing %s", pkg_group)
                continue
        updated_lines.append(line)
    tex_path.write_text("\n".join(updated_lines), encoding="utf-8")
    return True


__all__ = ["optimize_preamble"]
