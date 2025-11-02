from __future__ import annotations
from pathlib import Path
from typing import Iterable, Optional, Tuple

REQUIRED_INDEX_FILES = ("faiss.index", "faiss.meta.json")

def has_index(dir_path: Path) -> bool:
    return all((dir_path / f).exists() for f in REQUIRED_INDEX_FILES)

def find_first_with_index(candidates: Iterable[Path]) -> Optional[Path]:
    for p in candidates:
        if has_index(p):
            return p
    return None

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def link_or_copy(src: Path, dst: Path) -> Tuple[str, str]:
    try:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src)
        return ("linked", str(dst))
    except Exception:
        dst.write_bytes(src.read_bytes())
        return ("copied", str(dst))
