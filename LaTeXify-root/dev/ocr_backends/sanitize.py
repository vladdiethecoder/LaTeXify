# -*- coding: utf-8 -*-
"""
Sanitizers for OCR model outputs (Qwen2-VL, etc.).

- Strips ChatML/vision special tokens that sometimes leak even with
  skip_special_tokens=True (defensive).
- Optionally removes hallucinated TikZ environments.
- Deduplicates consecutive identical lines.
"""
from __future__ import annotations
import re
from typing import Iterable

# Common Qwen ChatML & vision tokens that can leak in outputs
_QWEN_TOKENS = [
    r"<\|im_start\|>", r"<\|im_end\|>",
    r"<\|vision_start\|>", r"<\|vision_end\|>",
    r"<\|image_pad\|>", r"<\|video_pad\|>",
    r"<im_start>", r"<im_end>",
]
_QWEN_TOKEN_RX = re.compile("|".join(_QWEN_TOKENS))

_TIKZ_ENV_RX = re.compile(
    r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}",
    flags=re.DOTALL | re.IGNORECASE,
)

def _dedup_adjacent_lines(text: str) -> str:
    out, last = [], None
    for ln in text.splitlines():
        if ln == last:
            continue
        out.append(ln)
        last = ln
    return "\n".join(out)

def strip_special_tokens(text: str) -> str:
    t = _QWEN_TOKEN_RX.sub("", text)
    # strip stray code fences that models sometimes inject
    t = re.sub(r"```+(\w+)?", "", t)
    # normalize whitespace
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def remove_tikz_envs(text: str) -> str:
    return _TIKZ_ENV_RX.sub("", text)

def sanitize_output(text: str, remove_tikz: bool = True, dedup_lines: bool = True) -> str:
    t = strip_special_tokens(text)
    if remove_tikz:
        t = remove_tikz_envs(t)
    if dedup_lines:
        t = _dedup_adjacent_lines(t)
    return t

def sanitize_batch(texts: Iterable[str], **kwargs) -> list[str]:
    return [sanitize_output(t, **kwargs) for t in texts]
