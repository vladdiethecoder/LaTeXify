#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model_router.py

Content-aware routing for LaTeX synthesis:
- math-heavy → CodeLlama-70B-Instruct (GGUF)
- tables/figures → Mixtral-8x7B-Instruct (GGUF)
- fallback → first available GGUF under provided search roots

Search roots include the two directories you provided plus optional HF cache.
If multiple .gguf exist, we prefer higher-fidelity quantizations (q5_k_m > q4_k_m > q5 > q4 > q3 > q2).
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
import json
import os
import re

# Default search roots (edit if needed)
DEFAULT_SEARCH_DIRS = [
    Path("/run/media/vdubrov/Active Storage/huggingface_cache/hub/models--TheBloke--CodeLlama-70B-Instruct-GGUF/"),
    Path("/run/media/vdubrov/Active Storage/huggingface_cache/hub/models--TheBloke--Mixtral-8x7B-Instruct-v0.1-GGUF/"),
]

# If HF_HOME or --hf-cache provided, include it
def _expand_search_dirs(extra_root: Optional[Path]) -> List[Path]:
    roots = list(DEFAULT_SEARCH_DIRS)
    hf_home = os.environ.get("HF_HOME") or os.environ.get("HF_DATASETS_CACHE")
    for p in [extra_root, Path(hf_home) if hf_home else None]:
        if p:
            roots.append(Path(p))
    return [p for p in roots if p and p.exists()]

QUANT_PREF = [
    "q5_k_m", "q5_k_l", "q5_k_s",
    "q4_k_m", "q4_k_s",
    "q5", "q4", "q3", "q2",
]

def _quant_score(name: str) -> int:
    lname = name.lower()
    for i, q in enumerate(QUANT_PREF):
        if q in lname:
            return len(QUANT_PREF) - i
    return 0

def find_gguf(root: Path, prefer_hint: Optional[str] = None) -> Optional[Path]:
    """
    Recursively find a .gguf under `root`. Prefer filenames containing `prefer_hint`
    and better quantization (by QUANT_PREF order).
    """
    if not root.exists():
        return None
    candidates: List[Tuple[int, Path]] = []
    for p in root.rglob("*.gguf"):
        score = _quant_score(p.name)
        if prefer_hint and prefer_hint.lower() in p.name.lower():
            score += 100
        candidates.append((score, p))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]

@dataclass
class RouteDecision:
    reason: str
    model_path: Path

MATH_PAT = re.compile(r"(\\begin\{align\*?\}|\\begin\{equation\*?\}|\\\[|\\\(|\$\$|\$[^$]+\$)")
TABLE_PAT = re.compile(r"(\\begin\{tabular|\\begin\{longtable|\\toprule|\\midrule|\\bottomrule)")
FIG_PAT = re.compile(r"(\\includegraphics|\\begin\{figure)")

def classify_bundle(text: str) -> str:
    """
    Rough, fast classification for routing.
    """
    is_math = bool(MATH_PAT.search(text))
    is_table = bool(TABLE_PAT.search(text))
    is_figure = bool(FIG_PAT.search(text))
    if is_math and not (is_table or is_figure):
        return "math"
    if is_table or is_figure:
        return "tables_figures"
    return "prose"

def choose_model_for_text(
    text: str,
    extra_search_root: Optional[Path] = None
) -> RouteDecision:
    roots = _expand_search_dirs(extra_search_root)
    c = classify_bundle(text)
    if c == "math":
        # Prefer CodeLlama-70B (math-leaning)
        for r in roots:
            if "CodeLlama-70B-Instruct-GGUF" in str(r):
                gg = find_gguf(r, prefer_hint="q5_k_m")
                if gg:
                    return RouteDecision("math→CodeLlama-70B", gg)
    if c == "tables_figures":
        # Prefer Mixtral-8x7B for structured/tabular content
        for r in roots:
            if "Mixtral-8x7B-Instruct" in str(r):
                gg = find_gguf(r, prefer_hint="q5_k_m")
                if gg:
                    return RouteDecision("tables/figures→Mixtral-8x7B", gg)
    # Fallback: first good GGUF anywhere
    for r in roots:
        gg = find_gguf(r, prefer_hint="q5_k_m")
        if gg:
            return RouteDecision("fallback→first-available", gg)
    raise FileNotFoundError("No .gguf model found in search roots.")
