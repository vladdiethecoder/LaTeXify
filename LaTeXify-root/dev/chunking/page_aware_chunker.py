# dev/chunking/page_aware_chunker.py
from __future__ import annotations

import io
import json
import os
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
from PIL import Image


# ====== Public API ======

@dataclass
class Chunk:
    id: str
    page: int
    label: str                     # e.g., TEXT, TITLE, LIST, TABLE, FIGURE
    text: str
    source_image: str              # path to page image
    ocr_model: str                 # which OCR file we used (if known)
    bbox: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2) in pixel coords
    meta: Optional[Dict[str, Any]] = None             # freeform

    def to_json(self) -> str:
        rec = asdict(self)
        return json.dumps(rec, ensure_ascii=False)


def build_chunks_for_run(
    run_dir: Path,
    pdf_path: Path,
    max_chars: int = 800,
    overlap: int = 120,
    min_par_len: int = 60,
) -> List[Chunk]:
    """
    Create page-aware chunks by:
      1) selecting OCR text per page (pref: nanonets-ocr2-3b → nanonets-ocr-s → qwen2...),
      2) calling HURIDOCS layout API on each page image (HTTP),
      3) mapping OCR text to layout blocks,
      4) producing length-bounded overlapping chunks.

    Falls back to simple, heuristic paragraphs if the API is not reachable.
    Returns the full list of chunks and ALSO writes dev/runs/<stamp>/chunks.jsonl.
    """
    run_dir = Path(run_dir)
    pages_dir = run_dir / "pages"
    if not pages_dir.exists():
        raise FileNotFoundError(f"No page images in {pages_dir}")

    # Enumerate page images (sorted lexicographically -> page-0001.png, ...)
    page_images = sorted(pages_dir.glob("page-*.png"))
    if not page_images:
        raise FileNotFoundError("No page images found (expected page-*.png).")

    # Load outputs/*/<page>.md so we can pick best OCR per page
    ocr_map = _index_ocr_outputs(run_dir)

    # Prepare HURIDOCS endpoint/config
    api_url = os.getenv("HURIDOCS_API_URL", "http://127.0.0.1:5060")
    api_key = os.getenv("HURIDOCS_API_KEY")  # optional; service may be unauthenticated locally

    all_chunks: List[Chunk] = []
    for idx, img_path in enumerate(page_images, start=1):
        page_no = _page_number_from_name(img_path.name) or idx
        ocr_model, page_text = _select_page_text(ocr_map, img_path.name)

        # Try layout over HTTP, then fallback
        blocks = None
        try:
            blocks = _huridocs_analyze_page(api_url, api_key, img_path)
        except Exception as e:
            print(f"[chunker] Layout API error on {img_path.name}: {e} — using heuristic fallback.")

        if not blocks:
            # Make a single TEXT block spanning the page; heuristics will chunk the text.
            blocks = [{
                "label": "TEXT",
                "bbox": None,
                "lines": _split_lines(page_text),
            }]

        # Convert blocks + OCR text into chunks, bounded by max_chars/overlap
        page_chunks = _blocks_to_chunks(
            blocks=blocks,
            page_text=page_text,
            page_no=page_no,
            img_path=img_path,
            ocr_model=ocr_model,
            max_chars=max_chars,
            overlap=overlap,
            min_par_len=min_par_len,
        )
        all_chunks.extend(page_chunks)

    # Write chunks.jsonl at run root
    out_path = run_dir / "chunks.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for ch in all_chunks:
            f.write(ch.to_json() + "\n")
    print(f"Wrote {out_path} with {len(all_chunks)} chunks")

    return all_chunks


# ====== HURIDOCS HTTP client ======

def _huridocs_analyze_page(
    api_url: str,
    api_key: Optional[str],
    page_image: Path,
    timeout: int = 60,
) -> List[Dict[str, Any]]:
    """
    Calls HURIDOCS PDF Document Layout Analysis API (VGT-backed) for a single *image* page.
    Expected response (per their examples) is a list of blocks with at least:
      - label: str (TEXT, TITLE, LIST, TABLE, FIGURE, etc.)
      - bbox: [x1, y1, x2, y2]  (pixel coordinates)
      - lines: [ "text line 1", "text line 2", ... ]    (optional; we’ll be robust if absent)

    This function accepts images because our pipeline has already rasterized the PDF.
    """
    url = _join_url(api_url, "/analyze")
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # Read image bytes
    with open(page_image, "rb") as fp:
        files = {"file": (page_image.name, fp, "image/png")}
        resp = requests.post(url, files=files, headers=headers, timeout=timeout)

    if resp.status_code != 200:
        raise RuntimeError(f"HURIDOCS API {url} returned {resp.status_code}: {resp.text[:300]}")

    data = resp.json()
    # Normalize: some deployments wrap blocks under {"blocks": [...]}
    blocks = data.get("blocks") if isinstance(data, dict) else data
    if not isinstance(blocks, list):
        raise ValueError("Unexpected HURIDOCS response format; expected a list of blocks or {'blocks': [...]}")

    # Defensive: ensure minimal keys exist
    norm: List[Dict[str, Any]] = []
    for b in blocks:
        label = b.get("label") or "TEXT"
        bbox = b.get("bbox")  # list of four ints/floats
        lines = b.get("lines") or []
        if isinstance(lines, str):
            lines = _split_lines(lines)
        norm.append({"label": label, "bbox": _coerce_bbox(bbox), "lines": lines})
    return norm


def _join_url(base: str, path: str) -> str:
    if base.endswith("/"):
        base = base[:-1]
    return f"{base}{path}"


def _coerce_bbox(b: Any) -> Optional[Tuple[int, int, int, int]]:
    if not b or not isinstance(b, (list, tuple)) or len(b) != 4:
        return None
    try:
        x1, y1, x2, y2 = b
        return (int(x1), int(y1), int(x2), int(y2))
    except Exception:
        return None


# ====== OCR outputs + text utilities ======

def _index_ocr_outputs(run_dir: Path) -> Dict[str, Dict[str, Path]]:
    """
    Returns mapping: { page_name -> { model_name -> md_path } }
    where md_path is dev/runs/<stamp>/outputs/<model>/<page>.md
    """
    out_dir = run_dir / "outputs"
    per_page: Dict[str, Dict[str, Path]] = {}
    if not out_dir.exists():
        return per_page

    for model_dir in out_dir.iterdir():
        if not model_dir.is_dir():
            continue
        model = model_dir.name
        for md_path in model_dir.glob("page-*.md"):
            per_page.setdefault(md_path.name.replace(".md", ".png"), {})[model] = md_path
    return per_page


def _select_page_text(ocr_map: Dict[str, Dict[str, Path]], page_png_name: str) -> Tuple[str, str]:
    """
    Preference order: nanonets-ocr2-3b → nanonets-ocr-s → qwen2-vl-ocr-2b-instruct → anything available.
    Returns (model_used, text_md).
    """
    pref = ["nanonets-ocr2-3b", "nanonets-ocr-s", "qwen2-vl-ocr-2b-instruct"]
    models = ocr_map.get(page_png_name, {})
    for m in pref:
        if m in models:
            return m, models[m].read_text(encoding="utf-8")
    # fallback: first available
    if models:
        m, p = next(iter(models.items()))
        return m, p.read_text(encoding="utf-8")
    # last resort: empty
    return "unknown", ""


def _split_lines(text: str) -> List[str]:
    # Normalize newlines and split
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return [ln.strip() for ln in text.split("\n")]


def _simple_paragraphs(text: str) -> List[str]:
    """
    Heuristic paragraph splitter: blank-line separation with a guard to merge tiny fragments.
    """
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    parts = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    # Merge very small parts with neighbors
    merged: List[str] = []
    buf = ""
    for p in parts:
        if len(p) < 60:
            buf = (buf + "\n\n" + p).strip()
        else:
            if buf:
                merged.append(buf)
                buf = ""
            merged.append(p)
    if buf:
        merged.append(buf)
    return merged


# ====== Chunking helpers ======

def _blocks_to_chunks(
    blocks: List[Dict[str, Any]],
    page_text: str,
    page_no: int,
    img_path: Path,
    ocr_model: str,
    max_chars: int,
    overlap: int,
    min_par_len: int,
) -> List[Chunk]:
    """
    Converts HURIDOCS blocks + OCR text to overlapped chunks.
    If a block has "lines", we join them; otherwise we fallback to paragraph heuristics on the full page text.
    """
    chunks: List[Chunk] = []
    base_id = f"p{page_no:04d}"

    if not blocks or not any((b.get("lines") for b in blocks)):
        # No structured lines → apply paragraph heuristics on whole page text
        paragraphs = _simple_paragraphs(page_text)
        seq = 0
        for para in paragraphs:
            for sub in _window_text(para, max_chars, overlap):
                chunks.append(Chunk(
                    id=f"{base_id}-h{seq:03d}",
                    page=page_no,
                    label="TEXT",
                    text=sub,
                    source_image=str(img_path),
                    ocr_model=ocr_model,
                    bbox=None,
                    meta={"strategy": "heuristic"},
                ))
                seq += 1
        return chunks

    # Structured: one chunk window per block (respecting min_par_len + windows)
    seq = 0
    for b in blocks:
        label = b.get("label", "TEXT")
        bbox = _coerce_bbox(b.get("bbox"))
        lines = b.get("lines") or []
        block_text = "\n".join(l for l in lines if isinstance(l, str)).strip()

        if not block_text:
            continue

        # If block too short, we may merge neighboring small blocks of the same label;
        # here we keep it simple and just skip sub-min blocks, unless it’s TITLE.
        if label != "TITLE" and len(block_text) < min_par_len:
            continue

        for sub in _window_text(block_text, max_chars, overlap):
            chunks.append(Chunk(
                id=f"{base_id}-b{seq:03d}",
                page=page_no,
                label=label,
                text=sub,
                source_image=str(img_path),
                ocr_model=ocr_model,
                bbox=bbox,
                meta={"strategy": "layout"},
            ))
            seq += 1

    # If nothing survived filters, ensure we still emit something
    if not chunks and page_text.strip():
        for i, sub in enumerate(_window_text(page_text, max_chars, overlap)):
            chunks.append(Chunk(
                id=f"{base_id}-f{i:03d}",
                page=page_no,
                label="TEXT",
                text=sub,
                source_image=str(img_path),
                ocr_model=ocr_model,
                bbox=None,
                meta={"strategy": "fallback-page"},
            ))
    return chunks


def _window_text(text: str, max_chars: int, overlap: int) -> Iterable[str]:
    """
    Sliding window over characters with overlap; respects word boundaries where possible.
    """
    text = " ".join(text.split())  # collapse whitespace
    if not text:
        return []
    if max_chars <= 0:
        yield text
        return

    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        # try to end on a word boundary (space/punct) without going backwards too far
        if end < n:
            back = text.rfind(" ", start + int(max_chars * 0.6), end)
            if back > start:
                end = back
        yield text[start:end].strip()
        if end == n:
            break
        start = max(0, end - overlap)


def _page_number_from_name(name: str) -> Optional[int]:
    m = re.search(r"page-(\d+)\.png$", name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None
