"""Auto-generate BibTeX entries from detected citations via LLM."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, List

from ..models.vllm_client import get_vllm_client

LOGGER = logging.getLogger(__name__)
CITE_RE = re.compile(r"\\cite[tp]?\{([^}]+)\}")
PROMPT = """You generate BibTeX entries for the cited works. \
Return a JSON object mapping each citation key to a BibTeX entry string. \
If you do not recognize a key, create a plausible placeholder entry with title and year.

Citation keys: {keys}

Context:
{context}
"""
BIBLIOGRAPHY_TAG = "latexify_auto"
STYLE_LINE = "\\bibliographystyle{plain}"
BIB_LINE = f"\\bibliography{{{BIBLIOGRAPHY_TAG}}}"


def _placeholder_entry(key: str) -> str:
    return (
        f"@misc{{{key},\n"
        "  title={Placeholder reference},\n"
        "  author={Unknown},\n"
        "  year={2024},\n"
        "  note={Auto-generated placeholder for missing citation}\n"
        "}"
    )


def _ensure_bibliography_hook(tex_path: Path, entries_created: bool) -> bool:
    if not entries_created:
        return False
    text = tex_path.read_text(encoding="utf-8")
    if BIB_LINE in text:
        return False
    insertion = f"{STYLE_LINE}\n{BIB_LINE}\n"
    if "\\end{document}" in text:
        text = text.replace("\\end{document}", insertion + "\\end{document}", 1)
    else:
        text = text + "\n" + insertion + "\n"
    tex_path.write_text(text, encoding="utf-8")
    return True


def generate_bibliography(tex_path: Path, output_path: Path) -> Dict[str, object]:
    report: Dict[str, object] = {"created": False, "count": 0, "output": str(output_path)}
    text = tex_path.read_text(encoding="utf-8")
    keys: List[str] = []
    for match in CITE_RE.finditer(text):
        for token in match.group(1).split(","):
            key = token.strip()
            if key and key not in keys:
                keys.append(key)
    if not keys:
        return report
    client = get_vllm_client()
    mapping: Dict[str, str] = {}
    if client is not None:
        excerpt = text[:8000]
        prompt = PROMPT.format(keys=", ".join(keys), context=excerpt)
        try:  # pragma: no cover - depends on vLLM runtime
            response = client.generate(prompt, stop=[], max_tokens=768).strip()
            if response.lower().startswith("<latex>"):
                response = response.split("</latex>", 1)[0]
            data = json.loads(response)
            if isinstance(data, dict):
                for key, entry in data.items():
                    if isinstance(entry, str) and entry.strip():
                        mapping[key.strip()] = entry.strip()
        except Exception as exc:
            LOGGER.debug("Auto-bibliography LLM failed: %s", exc)
    entries: List[str] = []
    for key in keys:
        entry = mapping.get(key)
        if not entry:
            entry = _placeholder_entry(key)
        entries.append(entry.strip())
    if not entries:
        return report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n\n".join(entries) + "\n", encoding="utf-8")
    hook = _ensure_bibliography_hook(tex_path, True)
    report.update({"created": True, "count": len(entries), "hook_inserted": hook})
    return report


__all__ = ["generate_bibliography"]
