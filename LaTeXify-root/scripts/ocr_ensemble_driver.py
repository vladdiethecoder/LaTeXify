#!/usr/bin/env python3
"""
ocr_ensemble_driver.py (PYTHONPATH-safe, overwrite-friendly)

- Imports each backend's `Backend` and calls `recognize_page(image_path, page=...)`.
- Renders PDF → PNG via PyMuPDF.
- Writes: <run_dir>/outputs/<slug>/page-0001.md (+ .json)
- New: --clean_outputs will remove an existing backend output dir first.

Backends:
  - nanonets_ocr2   → nanonets-ocr2-3b
  - nanonets_s      → nanonets-ocr-s
  - qwen2vl_ocr2b   → qwen2-vl-ocr-2b
"""
from __future__ import annotations

import argparse
import base64
import importlib
import json
import shutil
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import fitz  # PyMuPDF

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BACKENDS: List[Tuple[str, str, str]] = [
    ("nanonets_ocr2", "nanonets-ocr2-3b", "Nanonets OCR2 3B"),
    ("nanonets_s",    "nanonets-ocr-s",   "Nanonets Small"),
    ("qwen2vl_ocr2b", "qwen2-vl-ocr-2b",  "Qwen2-VL OCR 2B"),
]


def _rasterize(pdf: Path, out_dir: Path, dpi: int) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(str(pdf))
    scale = dpi / 72.0
    mat = fitz.Matrix(scale, scale)
    pages: List[Path] = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=mat, alpha=False)
        p = out_dir / f"page-{i+1:04d}.png"
        pix.save(str(p))
        pages.append(p)
    doc.close()
    return pages


def _safe_asdict(obj):
    if is_dataclass(obj):
        return asdict(obj)
    d = {}
    for key in ("model", "page", "text_md", "blocks"):
        if hasattr(obj, key):
            d[key] = getattr(obj, key)
    return d


def _persist_block_assets(
    blocks: Iterable[dict] | None,
    assets_dir: Path,
    backend_slug: str,
    page_idx: int,
    base_dir: Path,
) -> None:
    if not blocks:
        return
    assets_dir.mkdir(parents=True, exist_ok=True)
    for asset_idx, block in enumerate(blocks, start=1):
        btype = str(block.get("type") or block.get("block_type") or "").lower()
        if btype not in {"figure", "table"}:
            continue
        asset_path = block.get("asset_path") or block.get("image_path") or block.get("path")
        data = block.get("image_data") or block.get("image_bytes")
        ext = block.get("extension") or block.get("format")
        if isinstance(asset_path, str):
            candidate = Path(asset_path)
            search_paths = [candidate]
            if not candidate.is_absolute():
                search_paths.extend([
                    base_dir / candidate,
                    base_dir.parent / candidate,
                    candidate.resolve(),
                ])
            for path in search_paths:
                if path.exists():
                    suffix = path.suffix or (f".{ext}" if isinstance(ext, str) and ext else "")
                    target = assets_dir / f"{backend_slug}-p{page_idx:04d}-asset-{asset_idx:02d}{suffix}"
                    shutil.copy2(path, target)
                    break
            else:
                candidate = None
            if candidate is not None and any(path.exists() for path in search_paths):
                continue
        if isinstance(data, str):
            try:
                payload = base64.b64decode(data)
            except Exception:
                continue
            extension = (f".{ext}" if isinstance(ext, str) and ext else ".bin")
            target = assets_dir / f"{backend_slug}-p{page_idx:04d}-asset-{asset_idx:02d}{extension}"
            with target.open("wb") as handle:
                handle.write(payload)


def _write_page(out_dir: Path, page_idx: int, result_obj, *, assets_dir: Path | None, backend_slug: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = _safe_asdict(result_obj)
    text = meta.get("text_md") or ""
    (out_dir / f"page-{page_idx:04d}.md").write_text(text, encoding="utf-8")
    (out_dir / f"page-{page_idx:04d}.json").write_text(
        json.dumps({k: v for k, v in meta.items() if k in ("model", "page")}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if assets_dir is not None:
        blocks = meta.get("blocks") if isinstance(meta, dict) else getattr(result_obj, "blocks", None)
        if isinstance(blocks, list):
            _persist_block_assets(blocks, assets_dir, backend_slug, page_idx, out_dir)


def _load_backend(module_name: str):
    mod = importlib.import_module(f"dev.ocr_backends.{module_name}")
    if not hasattr(mod, "Backend"):
        raise RuntimeError(f"{module_name} has no Backend class")
    return mod.Backend()  # type: ignore


def main() -> None:
    ap = argparse.ArgumentParser(description="Run OCR ensemble and write page-*.md per backend")
    ap.add_argument("--pdf", type=Path, required=True)
    ap.add_argument("--run_dir", type=Path, required=True)
    ap.add_argument("--only", type=str, default="")
    ap.add_argument("--dpi", type=int, default=180)
    ap.add_argument("--clean_outputs", action="store_true", help="Remove existing outputs/<slug> before writing.")
    ap.add_argument(
        "--assets-dir",
        type=Path,
        default=Path("build/assets"),
        help="Directory where detected figure/table crops will be exported.",
    )
    args = ap.parse_args()

    tmp_img_dir = args.run_dir.resolve() / "tmp" / "images"
    pages = _rasterize(args.pdf, tmp_img_dir, dpi=args.dpi)
    print(f"[raster] {len(pages)} pages at {args.dpi} dpi → {tmp_img_dir}")

    only = {s.strip() for s in args.only.split(",")} if args.only else set()

    for module_name, slug, label in BACKENDS:
        if only and slug not in only:
            continue

        try:
            backend = _load_backend(module_name)
        except Exception as e:
            print(f"[warn] {label}: could not instantiate ({e}). Skipping.", file=sys.stderr)
            continue

        out_dir = args.run_dir / "outputs" / slug
        if args.clean_outputs and out_dir.exists():
            shutil.rmtree(out_dir, ignore_errors=True)

        ok_pages = 0
        for i, img_path in enumerate(pages, start=1):
            try:
                result = backend.recognize_page(str(img_path), page=i)
                _write_page(out_dir, i, result, assets_dir=args.assets_dir, backend_slug=slug)
                ok_pages += 1
            except Exception as e:
                print(f"[warn] {label}: page {i} failed ({e})", file=sys.stderr)

        print(f"[{label}] wrote {ok_pages} pages → {out_dir}")

    print("[ocr-ensemble] done.")


if __name__ == "__main__":
    main()
