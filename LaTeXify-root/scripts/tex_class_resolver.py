#!/usr/bin/env python3
"""
tex_class_resolver.py
DCL-aware TeX class discovery + optional stub creation.

Search order (highest priority first):
  1) kb/offline/** (exact match directory first)
  2) kb/online/**  (exact match directory first)
  3) LiX/          (fallback)

If <doc_class>.cls is not found, we can create a tiny stub class that simply
\LoadClass{article} so compilation succeeds. The stub directory is then
prepended to TEXINPUTS.

This module only manages paths/files. Exporting TEXINPUTS and compiling is done
by the pipeline.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

STUB_HEADER = r"""\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{%(doc_class)s}[2025/10/30 LiX fallback stub]
\LoadClass{article}
"""

def _search_exact(root: Path, doc_class: str) -> Optional[Path]:
    """Return the directory containing <doc_class>.cls under root/** if present."""
    target = f"{doc_class}.cls"
    if not root.exists():
        return None
    for hit in root.rglob("*.cls"):
        if hit.name == target:
            return hit.parent
    return None

def discover_dirs_dcl(root: Path, doc_class: str) -> Tuple[List[Path], Optional[Path]]:
    """
    Return (ordered_dirs, exact_dir_match) where ordered_dirs is the DCL path list.
    """
    root = root.resolve()
    kb_off = root / "kb" / "offline"
    kb_on  = root / "kb" / "online"
    lix    = root / "LiX"

    exact_off = _search_exact(kb_off, doc_class)
    exact_on  = _search_exact(kb_on,  doc_class)

    ordered: List[Path] = []
    if exact_off: ordered.append(exact_off)
    # canonical offline buckets (existence checked later)
    ordered += [kb_off / "classes", kb_off / "latex", kb_off / "course"]

    if exact_on: ordered.append(exact_on)
    # canonical online buckets
    ordered += [kb_on / "classes", kb_on / "latex", kb_on / "course"]

    ordered.append(lix)

    # de-duplicate while preserving order and existing dirs only
    seen, dedup = set(), []
    for d in ordered:
        if d.exists():
            rd = d.resolve()
            if rd not in seen:
                seen.add(rd)
                dedup.append(rd)
    return dedup, (exact_off or exact_on)

def ensure_stub_if_missing(root: Path, build_dir: Path, doc_class: str,
                           dirs: List[Path], exact_dir: Optional[Path]) -> Tuple[List[Path], Optional[Path], bool]:
    """
    If <doc_class>.cls is not found in any of 'dirs', create a stub under build/_stubs
    and return (dirs_with_stub_first, stub_dir, created_flag).
    """
    target = f"{doc_class}.cls"
    # Fast check: present anywhere?
    for d in dirs:
        if (d / target).exists():
            return dirs, None, False

    # Create stub
    stub_dir = (build_dir / "_stubs").resolve()
    stub_dir.mkdir(parents=True, exist_ok=True)
    (stub_dir / target).write_text(STUB_HEADER % {"doc_class": doc_class}, encoding="utf-8")

    # Prepend stub_dir to search order
    new_dirs = [stub_dir] + dirs
    return new_dirs, stub_dir, True

def texinputs_from_dirs(dirs: List[Path], prepend: str = "") -> str:
    """Build TEXINPUTS string with trailing ':' to include default Kpathsea paths."""
    parts = [p.as_posix() for p in dirs]
    s = ":".join(parts) + ":"
    return f"{prepend}{s}" if prepend else s
