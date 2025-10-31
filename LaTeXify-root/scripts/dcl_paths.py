#!/usr/bin/env python3
"""
dcl_paths.py (revised)
Legacy helper kept for backward compatibility; delegates to tex_class_resolver.

Prefer importing from `tex_class_resolver` in new code.
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional

from tex_class_resolver import discover_dirs_dcl, texinputs_from_dirs  # re-export

def discover_tex_dirs(root: Path, doc_class: str) -> List[Path]:
    dirs, _ = discover_dirs_dcl(root, doc_class)
    return dirs

def build_env_with_texinputs(env: dict, dirs: List[Path]) -> tuple[dict, str]:
    new_env = env.copy()
    s = texinputs_from_dirs(dirs)
    new_env["TEXINPUTS"] = s + new_env.get("TEXINPUTS", "")
    return new_env, s
