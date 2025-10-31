#!/usr/bin/env python3
"""
lix_fixup.py — Detect & fix LiX class installs (no network).

- Detects if <doc_class>.cls resolved by kpsewhich is a Git LFS pointer.
- Copies *real* LiX classes from --source into TEXMFHOME/tex/latex/lix,
  skipping LFS pointers, dangling symlinks, and unexpected filenames.
- Overwrites existing files (including prior LFS pointers).
- Emits compact JSON events to stdout; exits 0 on success, nonzero on hard fail.

Deterministic; Python 3.11+.
"""
from __future__ import annotations
import argparse, json, os, re, shutil, subprocess, sys
from pathlib import Path
from typing import Iterable, List

# Known LiX classes we allow to install; avoids odd files like "Link to novel.cls"
ALLOWED_CLASSES = {
    "contract", "ieee_modern", "news", "novel", "novella",
    "paper", "poem", "textbook", "thesis",
}

LFS_RE = re.compile(r"^\s*version\s+https://git-lfs\.github\.com/spec/v1\s*$")

def _json(event: str, **details):
    print(json.dumps({"event": event, **details}, ensure_ascii=False))

def _run(cmd: List[str]) -> tuple[int, str]:
    p = subprocess.run(cmd, text=True, capture_output=True)
    return p.returncode, (p.stdout or "").strip()

def kpsewhich_file(name: str) -> Path | None:
    rc, out = _run(["kpsewhich", name])
    if rc == 0 and out:
        p = Path(out)
        return p if p.exists() else None
    return None

def is_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                if i > 2:
                    break
                if LFS_RE.match(line):
                    return True
        return False
    except Exception:
        return False

def default_texmfhome() -> Path:
    env = os.environ.get("TEXMFHOME")
    if env:
        return Path(env).expanduser().resolve()
    rc, out = _run(["kpsewhich", "-var-value=TEXMFHOME"])
    if rc == 0 and out:
        return Path(out).expanduser().resolve()
    return Path.home() / "texmf"

def is_regular_file(p: Path) -> bool:
    # True for ordinary files (follows symlinks only if they resolve)
    try:
        if p.is_symlink() and not p.exists():
            return False
        return p.is_file()
    except Exception:
        return False

def safe_copy(src: Path, dst: Path) -> bool:
    try:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        shutil.copy2(src, dst)
        return True
    except Exception as e:
        _json("copy_skip", src=str(src), dst=str(dst), reason=str(e))
        return False

def iter_real_lix_files(src: Path) -> Iterable[Path]:
    # yield real .cls (from ALLOWED set) and lix.sty only
    for f in src.iterdir():
        if f.name == "lix.sty":
            if is_regular_file(f) and not is_lfs_pointer(f):
                yield f
            else:
                _json("skip_bad_file", path=str(f))
            continue
        if f.suffix == ".cls" and f.stem in ALLOWED_CLASSES:
            if is_regular_file(f) and not is_lfs_pointer(f):
                yield f
            else:
                _json("skip_bad_file", path=str(f))
            continue
        # ignore everything else silently (e.g., “Link to novel.cls”)
    return

def install_lix(source: Path, texmfhome: Path) -> dict:
    dest = texmfhome / "tex" / "latex" / "lix"
    dest.mkdir(parents=True, exist_ok=True)
    copied, skipped = [], []
    any_candidates = False
    for f in iter_real_lix_files(source):
        any_candidates = True
        target = dest / f.name
        if safe_copy(f, target):
            copied.append(str(target))
        else:
            skipped.append(f.name)
    return {"dest": str(dest), "copied": copied, "skipped": skipped, "had_candidates": any_candidates}

def main():
    ap = argparse.ArgumentParser(description="Fix LiX class install (detect LFS pointers; install real classes).")
    ap.add_argument("--doc_class", default="textbook", help="LiX doc class name (e.g., textbook, paper, novel)")
    ap.add_argument("--source", type=Path, required=False, help="Dir with real LiX *.cls + lix.sty")
    ap.add_argument("--install", action="store_true", help="Copy from --source into TEXMFHOME")
    args = ap.parse_args()

    cls_name = f"{args.doc_class}.cls"
    cls_path = kpsewhich_file(cls_name)
    if not cls_path:
        _json("class_missing", doc_class=args.doc_class, hint="Provide --source and --install to add classes.")
    else:
        _json("class_found", path=str(cls_path))
        if is_lfs_pointer(cls_path):
            _json("lfs_pointer_detected", path=str(cls_path))
        else:
            _json("ok_real_class", path=str(cls_path))
            # nothing to do
            sys.exit(0)

    if not args.install or not args.source:
        _json("action_required", hint="Re-run with --source <dir> --install to replace pointers.")
        sys.exit(1)

    src = args.source
    if not src.exists() or not src.is_dir():
        _json("bad_source", path=str(src))
        sys.exit(2)

    res = install_lix(src, default_texmfhome())
    _json("installed", **res)

    # re-check
    cls_path = kpsewhich_file(cls_name)
    ok = bool(cls_path) and not (cls_path and is_lfs_pointer(cls_path))
    _json("post_install_check", found=bool(cls_path), path=str(cls_path) if cls_path else None, is_pointer=(not ok))
    sys.exit(0 if ok else 3)

if __name__ == "__main__":
    main()
