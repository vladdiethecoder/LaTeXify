# dev/chunking/page_aware_chunker.py
from __future__ import annotations

import io
import json
import os
import re
import textwrap
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


# ---------------------------
# Public API
# ---------------------------

def build_chunks_for_run(
    run_dir: Path,
    pdf_path: Path,
    max_chars: int = 800,
    overlap: int = 120,
    min_par_len: int = 60,
) -> Path:
    """
    Build page-aware chunks for a run.
    • Tries HURIDOCS layout API (POST / with multipart 'file' = PDF).
    • Falls back to heuristic paragraph segmentation if API is absent.
    • Produces: <run_dir>/chunks.jsonl
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    pages_dir = run_dir / "pages"
    if not pages_dir.exists():
        raise FileNotFoundError(f"No page images in {pages_dir}")

    # Try HURIDOCS once per PDF (cache JSON).
    layout_cache_dir = run_dir / "layout"
    layout_cache_dir.mkdir(parents=True, exist_ok=True)
    layout_cache = layout_cache_dir / "huridocs_layout.json"

    layout = None
    try:
        layout = _fetch_layout_with_cache(pdf_path, layout_cache)
    except Exception as e:
        print(f"[chunker] Layout API error: {e} — using heuristic fallback.")

    # Load OCR outputs (pick best available per page)
    per_page_text: Dict[str, str] = _collect_page_texts(run_dir)

    # Turn layout into block-level texts where possible; otherwise heuristic
    records: List[Dict[str, Any]] = []
    pages = sorted(pages_dir.glob("page-*.png"))

    if layout:
        # Parse into a standard: Dict[page_index -> List[BlockCandidate]]
        parsed = _parse_huridocs_layout(layout)
        for page_idx, page_png in enumerate(pages, start=1):
            page_key = page_png.name
            page_text = per_page_text.get(page_key, "")
            blocks = parsed.get(page_idx, [])
            if not blocks:
                # Fallback for that page only
                records.extend(_heuristic_chunks(page_text, page_idx, page_png, max_chars, overlap, min_par_len))
            else:
                # Use blocks as paragraph-ish anchors; chunk inside them
                block_texts = _split_text_by_blocks(page_text, len(blocks)) if page_text else []
                for bi, bt in enumerate(block_texts or []):
                    for ch in _sliding_chunks(bt, max_chars, overlap, min_par_len):
                        records.append(_mk_rec(ch, page_idx, page_png, block_id=bi))
    else:
        # Global fallback
        for page_idx, page_png in enumerate(pages, start=1):
            page_key = page_png.name
            page_text = per_page_text.get(page_key, "")
            records.extend(_heuristic_chunks(page_text, page_idx, page_png, max_chars, overlap, min_par_len))

    # Write JSONL with stable IDs
    out_path = run_dir / "chunks.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for i, rec in enumerate(records):
            rec.setdefault("id", f"chunk_{i:06d}")
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {out_path} with {len(records)} chunks")
    return out_path


# ---------------------------
# Data structures
# ---------------------------

@dataclass
class BlockCandidate:
    bbox: Tuple[float, float, float, float]
    label: str = "block"
    score: Optional[float] = None


# ---------------------------
# HURIDOCS integration
# ---------------------------

def _fetch_layout_with_cache(pdf_path: Path, cache_path: Path) -> Optional[Dict[str, Any]]:
    """
    POST the PDF to HURIDOCS analyzer (default: POST /) and cache the JSON.
    Environment (all optional):
      HURIDOCS_API_URL  (default: http://127.0.0.1:5060)
      HURIDOCS_API_PATH (default: /)
      HURIDOCS_API_KEY  (Bearer token)
    """
    api_url = os.getenv("HURIDOCS_API_URL", "http://127.0.0.1:5060").rstrip("/")
    api_path = os.getenv("HURIDOCS_API_PATH", "/")  # Confirmed by your openapi.json
    url = api_url + api_path

    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            pass  # ignore cache errors

    headers = {}
    api_key = os.getenv("HURIDOCS_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    files = {"file": (pdf_path.name, pdf_path.open("rb"), "application/pdf")}
    resp = requests.post(url, headers=headers, files=files, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(
            f"HURIDOCS API {url} returned {resp.status_code}: {resp.text[:200]}"
        )

    data = resp.json()
    cache_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def _parse_huridocs_layout(data: Dict[str, Any]) -> Dict[int, List[BlockCandidate]]:
    """
    Tries several plausible shapes from the HURIDOCS service, returning:
      { page_index(int 1-based): [BlockCandidate, ...], ... }

    We’re intentionally permissive: if unknown, return {} and caller will fallback.
    """
    out: Dict[int, List[BlockCandidate]] = {}

    # Common shape A:
    # { "pages": [ { "page": 1, "blocks": [ { "bbox":[x0,y0,x1,y1], "label":"text", "score":0.98 }, ...] }, ...] }
    pages = data.get("pages")
    if isinstance(pages, list) and pages:
        for page_entry in pages:
            page_idx = page_entry.get("page") or page_entry.get("page_index") or page_entry.get("index")
            try:
                page_idx = int(page_idx)
            except Exception:
                continue
            blocks = []
            for b in page_entry.get("blocks", []):
                bbox = _extract_bbox(b)
                if bbox:
                    blocks.append(BlockCandidate(bbox=bbox, label=str(b.get("label", "block")), score=b.get("score")))
            if blocks:
                out[page_idx] = blocks

    if out:
        return out

    # Common shape B: flat list with page info on each block
    # { "blocks": [ { "page": 1, "bbox":[...], ...}, ...] }
    flat_blocks = data.get("blocks")
    if isinstance(flat_blocks, list) and flat_blocks:
        tmp: Dict[int, List[BlockCandidate]] = {}
        for b in flat_blocks:
            page_idx = b.get("page") or b.get("page_index") or b.get("index")
            bbox = _extract_bbox(b)
            if page_idx and bbox:
                try:
                    pg = int(page_idx)
                    tmp.setdefault(pg, []).append(
                        BlockCandidate(bbox=bbox, label=str(b.get("label", "block")), score=b.get("score"))
                    )
                except Exception:
                    pass
        if tmp:
            return tmp

    # Unknown shape
    return {}


def _extract_bbox(b: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    # Accept a variety of bbox field spellings
    bb = b.get("bbox") or b.get("box") or b.get("rect")
    if isinstance(bb, list) and len(bb) == 4:
        try:
            return (float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3]))
        except Exception:
            return None
    return None


# ---------------------------
# Heuristic & chunking helpers
# ---------------------------

def _collect_page_texts(run_dir: Path) -> Dict[str, str]:
    """
    Find per-page OCR text from the run’s outputs.
    Preference order (first match wins): nanonets-ocr2-3b, nanonets-ocr-s, qwen2-vl-ocr-2b-instruct, any.
    """
    outputs = run_dir / "outputs"
    if not outputs.exists():
        return {}
    model_order = ["nanonets-ocr2-3b", "nanonets-ocr-s", "qwen2-vl-ocr-2b-instruct"]
    # Collect all candidate files
    by_page: Dict[str, Dict[str, Path]] = {}
    for model_dir in outputs.glob("*"):
        if not model_dir.is_dir():
            continue
        model = model_dir.name
        for md in model_dir.glob("page-*.md"):
            by_page.setdefault(md.name, {})[model] = md

    page_texts: Dict[str, str] = {}
    for page, model_map in by_page.items():
        chosen = None
        for m in model_order:
            if m in model_map:
                chosen = model_map[m]
                break
        if not chosen:
            # pick any
            chosen = next(iter(model_map.values()))
        try:
            page_texts[page] = chosen.read_text(encoding="utf-8")
        except Exception:
            page_texts[page] = ""
    return page_texts


def _heuristic_chunks(
    page_text: str,
    page_idx: int,
    page_png: Path,
    max_chars: int,
    overlap: int,
    min_par_len: int,
) -> List[Dict[str, Any]]:
    paragraphs = _segment_paragraphs(page_text)
    recs: List[Dict[str, Any]] = []
    for pi, par in enumerate(paragraphs):
        for ch in _sliding_chunks(par, max_chars, overlap, min_par_len):
            recs.append(_mk_rec(ch, page_idx, page_png, block_id=pi))
    return recs


def _segment_paragraphs(text: str) -> List[str]:
    if not text:
        return []
    # Split on blank lines; merge short lines; keep bullets
    lines = [ln.strip() for ln in text.splitlines()]
    paras: List[str] = []
    buf: List[str] = []
    def _flush():
        if buf:
            paras.append(" ".join(buf).strip())
            buf.clear()
    for ln in lines:
        if not ln:
            _flush()
            continue
        if _is_bullet(ln):
            _flush()
            paras.append(ln)
        else:
            buf.append(ln)
    _flush()
    # Drop tiny paragraphs that are just noise
    return [p for p in paras if len(p) >= 10]


def _is_bullet(ln: str) -> bool:
    return ln.startswith(("-", "*", "\u2022")) or bool(re.match(r"^\d+[\.\)]\s", ln))


def _sliding_chunks(text: str, max_chars: int, overlap: int, min_par_len: int) -> List[str]:
    text = text.strip()
    if not text:
        return []
    # If already short, return as-is
    if len(text) <= max_chars:
        return [text] if len(text) >= min_par_len else []
    # Otherwise sliding window
    out: List[str] = []
    start = 0
    step = max_chars - overlap
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunk = text[start:end].strip()
        if len(chunk) >= min_par_len:
            out.append(chunk)
        if end == len(text):
            break
        start += step
    return out


def _split_text_by_blocks(full_text: str, n_blocks: int) -> List[str]:
    """
    Extremely simple splitter: divide the page text into N contiguous parts.
    A proper alignment would require OCR line coords; this keeps it robust.
    """
    if not full_text or n_blocks <= 0:
        return []
    avg = max(1, len(full_text) // n_blocks)
    parts = []
    i = 0
    for _ in range(n_blocks - 1):
        parts.append(full_text[i : i + avg])
        i += avg
    parts.append(full_text[i:])
    return [p.strip() for p in parts if p.strip()]


def _mk_rec(text: str, page_idx: int, page_png: Path, block_id: int) -> Dict[str, Any]:
    return {
        "text": text,
        "page": page_idx,
        "source_png": str(page_png),
        "block_id": block_id,
    }
