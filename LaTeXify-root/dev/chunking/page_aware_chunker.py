from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import requests

HURIDOCS_URL = "http://127.0.0.1:5060"


def _post_huridocs(pdf: Path, timeout_s: int = 30) -> dict | list:
    """POST the PDF to HURIDOCS `POST /` and return JSON."""
    with pdf.open("rb") as fh:
        r = requests.post(HURIDOCS_URL, files={"file": fh}, timeout=timeout_s)
    r.raise_for_status()
    return r.json()


def _fetch_layout_with_cache(
    pdf_path: Path, cache_path: Path, retries: int = 3, backoff: float = 2.0
):
    """Fetch layout once and cache the raw JSON to speed re-runs."""
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    last = None
    for a in range(retries):
        try:
            data = _post_huridocs(pdf_path)
            cache_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            return data
        except Exception as e:
            last = e
            time.sleep(backoff * (a + 1))
    if last:
        raise last


def _group_segments_by_page(segs: Iterable[dict]) -> List[dict]:
    """Group a flat list of segments into page dicts with `page_number` and `segments`."""
    pages: Dict[int, List[dict]] = {}
    for s in segs or []:
        pn = int(s.get("page_number") or s.get("page") or 1)
        pages.setdefault(pn, []).append(s)
    return [{"page_number": pn, "segments": lst} for pn, lst in sorted(pages.items())]


def _parse_huridocs_layout(data) -> Dict[int, List[dict]]:
    """Normalize HURIDOCS payloads into {page_index: [blocks...] }.

    Accepts:
      1) {"pages":[{"page_number":..,"blocks"| "segments":[...]}]}
      2) {"result":{"pages":[...]}}
      3) [{"page_number":..,"blocks"| "segments":[...]}]
      4) {"blocks"| "segments":[...]}  (flat list)
    """
    # normalize root → pages
    if isinstance(data, list):
        if data and isinstance(data[0], dict) and (
            ("left" in data[0]) or ("bbox" in data[0]) or ("text" in data[0])
        ):
            pages = _group_segments_by_page(data)
        else:
            pages = data
    elif isinstance(data, dict):
        if "pages" in data:
            pages = data["pages"]
        elif isinstance(data.get("result"), dict) and "pages" in data["result"]:
            pages = data["result"]["pages"]
        elif ("blocks" in data) or ("segments" in data):
            pages = _group_segments_by_page(
                data.get("blocks") or data.get("segments") or []
            )
        else:
            pages = []
    else:
        pages = []

    out: Dict[int, List[dict]] = {}
    for i, p in enumerate(pages, start=1):
        pn = int(p.get("page_number", i))
        blocks = p.get("blocks") or p.get("segments") or []
        norm = []
        for b in blocks:
            if isinstance(b.get("bbox"), (list, tuple)) and len(b["bbox"]) == 4:
                x0, y0, x1, y1 = b["bbox"]
            else:
                l = float(b.get("left", 0))
                t = float(b.get("top", 0))
                w = float(b.get("width", 0))
                h = float(b.get("height", 0))
                x0, y0, x1, y1 = l, t, l + w, t + h
            norm.append(
                {
                    "bbox": [x0, y0, x1, y1],
                    "label": (b.get("type") or b.get("label") or "paragraph").lower(),
                    "text": b.get("text") or "",
                }
            )
        out[pn] = norm
    return out


def _collect_page_texts(run_dir: Path) -> Dict[int, str]:
    """Return {page_index: plain_text} reading run_dir/outputs/_agg/page-####.txt."""
    agg = run_dir / "outputs" / "_agg"
    texts: Dict[int, str] = {}
    for p in sorted(agg.glob("page-*.txt")):
        try:
            idx = int(p.stem.split("-")[1])  # 'page-0001' -> 1
        except Exception:
            continue
        texts[idx] = p.read_text(encoding="utf-8")
    return texts


def _split_text_by_blocks(full_text: str, n_blocks: int) -> List[str]:
    """Split page text into N roughly equal parts (stable simple heuristic)."""
    if not full_text or n_blocks <= 0:
        return []
    avg = max(1, len(full_text) // n_blocks)
    parts, i = [], 0
    for _ in range(n_blocks - 1):
        parts.append(full_text[i : i + avg])
        i += avg
    parts.append(full_text[i:])
    return [p.strip() for p in parts if p.strip()]


def _sliding_chunks(text: str, max_chars: int, overlap: int, min_par_len: int) -> Iterable[str]:
    """Slide a fixed-size window (with overlap) over text; drop tiny segments."""
    text = " ".join(text.split())
    if not text:
        return []
    if len(text) <= max_chars:
        return [text] if len(text) >= min_par_len else []
    out, i = [], 0
    step = max(1, max_chars - overlap)
    while i < len(text):
        seg = text[i : i + max_chars]
        if len(seg) >= min_par_len:
            out.append(seg)
        i += step
    return out


def _mk_rec(text: str, page_idx: int, bbox=None, model="heuristic", conf=0.70) -> Dict[str, Any]:
    """Construct a chunk record matching chunks.jsonl schema."""
    return {
        "page": page_idx,
        "label": "paragraph",
        "bbox": bbox or [0, 0, 0, 0],
        "text": text,
        "model": model,
        "confidence": float(conf),
    }


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
    """Build page-aware chunks; uses HURIDOCS when available, else heuristic."""
    pages_dir = run_dir / "pages"
    if not pages_dir.exists():
        raise FileNotFoundError(f"No page images in {pages_dir} (render first)")

    # try layout (cached)
    layout_cache = run_dir / "layout"
    layout_cache.mkdir(exist_ok=True, parents=True)
    layout_file = layout_cache / "huridocs_layout.json"
    layout = None
    try:
        layout = _fetch_layout_with_cache(pdf_path, layout_file)
    except Exception as e:
        print(f"[chunker] layout API unavailable: {e} — using heuristic fallback.")

    per_page_text = _collect_page_texts(run_dir)
    pages = sorted(pages_dir.glob("page-*.png"))
    records: List[Dict[str, Any]] = []

    page_blocks = _parse_huridocs_layout(layout) if isinstance(layout, (dict, list)) else {}

    for page_idx, _png in enumerate(pages, start=1):
        text = per_page_text.get(page_idx, "")
        blocks = page_blocks.get(page_idx, [])
        if blocks:
            # distribute page text across blocks (simple & stable), then chunk
            block_texts = _split_text_by_blocks(text, len(blocks)) if text else [b.get("text", "") for b in blocks]
            for b, bt in zip(blocks, block_texts):
                for ch in _sliding_chunks(bt, max_chars, overlap, min_par_len):
                    records.append(_mk_rec(ch, page_idx, bbox=b.get("bbox"), model="huridocs+heur", conf=0.83))
        else:
            for ch in _sliding_chunks(text, max_chars, overlap, min_par_len):
                records.append(_mk_rec(ch, page_idx, model="heuristic", conf=0.70))

    out = run_dir / "chunks.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for i, r in enumerate(records):
            r.setdefault("id", f"ch_{i:06d}")
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[chunker] wrote {out} ({len(records)} chunks)")
    return out
