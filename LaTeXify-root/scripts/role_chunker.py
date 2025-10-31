#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Role-aware, blocks-based chunker with robust fallbacks.

This script implements the 4 role strategies:

- user_answers          (Hybrid Structural-by-question → Rolling inside the answer)
- assessment_questions  (Pure Structural: one chunk per question/sub-question)
- rubric_rows           (Fine-grained row clustering per rubric row/criterion)
- assignment_sections   (Sectional: headers as hard boundaries + rolling inside)

If blocks_refined.jsonl is missing or incomplete, it transparently falls back to
OCR page markdown + simple header anchors, then applies the same logic.

Outputs:
  <run_dir>/chunks.jsonl
  <run_dir>/chunks_meta.json

Each chunk json line minimally contains:
  {
    "id": "...",
    "role": "user|assessment|rubric|assignment",
    "question_id": "Q4" | "Q1a" | null,
    "criterion": "Align equals vertically" | null,
    "section": "AI Use Statement" | null,
    "page_span": [start_page, end_page],
    "block_ids": ["p004-b012", ...],
    "order": [first_page, first_block_order],
    "text": "..."
  }

Safe assumptions:
- blocks_refined.jsonl entries may vary in schema. We try to read:
    page (int), id (or block_id), bbox (or x0,y0,x1,y1), text (str),
    font_size (optional), bold/weight (optional), order index (optional).
- Page numbers may be 1-based or 0-based; we treat input as-is for ordering,
  and only use page numbers for sorting and metadata.
- If no blocks file exists, we synthesize pseudo-blocks from OCR page-*.md.

Heuristics are intentionally conservative so that chunk boundaries are stable,
atomic for questions, and DCL-friendly.

Author: ROG / LaTeXify
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# --------- Common helpers ---------

@dataclass
class Bx:
    page: int
    bid: str
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    font_size: float = 0.0
    weight: float = 0.0
    order: int = 0


def _read_blocks(path: Path) -> List[Bx]:
    """Best-effort parse of blocks_refined.jsonl (one json per line)."""
    blocks: List[Bx] = []
    if not path.exists():
        return blocks
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                j = json.loads(s)
            except Exception:
                continue
            page = int(j.get("page") or j.get("page_index") or j.get("p") or 0)
            bid = str(j.get("id") or j.get("block_id") or f"blk-{i:06d}")
            text = (j.get("text") or j.get("content") or "").strip()
            bbox = j.get("bbox")
            if bbox and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                x0, y0, x1, y1 = [float(v) for v in bbox]
            else:
                x0 = float(j.get("x0", 0.0))
                y0 = float(j.get("y0", 0.0))
                x1 = float(j.get("x1", 1.0))
                y1 = float(j.get("y1", 1.0))
            fs = float(j.get("font_size", j.get("fs", 0.0)) or 0.0)
            wt = float(j.get("weight", j.get("bold", 0.0)) or 0.0)
            order = int(j.get("order", j.get("block_idx", i)) or i)
            blocks.append(Bx(page=page, bid=bid, x0=x0, y0=y0, x1=x1, y1=y1,
                             text=text, font_size=fs, weight=wt, order=order))
    # sort blocks by (page, y0, x0) with order tie-break
    blocks.sort(key=lambda b: (b.page, b.y0, b.x0, b.order))
    return blocks


def _read_ocr_pages(run_dir: Path) -> List[Bx]:
    """Fallback: synthesize blocks from OCR page-*.md files."""
    outputs = run_dir / "outputs"
    pages: List[Bx] = []
    if not outputs.exists():
        return pages
    page_files = sorted(outputs.glob("*/page-*.md")) or sorted(outputs.glob("page-*.md"))
    # page index from filename: page-0007.md -> 7
    page_rx = re.compile(r"page-(\d+)")
    for i, p in enumerate(page_files, 1):
        m = page_rx.search(p.name)
        pg = int(m.group(1)) if m else i
        try:
            txt = p.read_text(encoding="utf-8").strip()
        except Exception:
            txt = ""
        # synthesize one big block per paragraph for minimal structure
        # split on blank lines or headings
        paragraphs = [seg.strip() for seg in re.split(r"(?ms)^\s*#{1,6}\s+.*?$|(?:\n\s*\n)+", txt) if seg.strip()]
        y = 0.0
        for j, pr in enumerate(paragraphs, 1):
            pages.append(Bx(page=pg, bid=f"p{pg:04d}-para{j:04d}",
                            x0=0.0, y0=y, x1=1.0, y1=y + 1.0,
                            text=pr, font_size=0.0, weight=0.0, order=j))
            y += 1.0
    pages.sort(key=lambda b: (b.page, b.order))
    return pages


def _window_text(text: str, max_chars: int, overlap: int) -> List[Tuple[int, int, str]]:
    """Slice long text into overlapping windows without breaking."""
    if not text:
        return []
    if len(text) <= max_chars:
        return [(0, len(text), text)]
    out: List[Tuple[int, int, str]] = []
    start = 0
    N = len(text)
    while start < N:
        end = min(N, start + max_chars)
        # avoid cutting immediately after a math delimiter if we can
        seg = text[start:end]
        out.append((start, end, seg))
        if end >= N:
            break
        start = max(0, end - overlap)
        if start >= N:
            break
    return out


def _join_blocks(blocks: List[Bx]) -> str:
    # join with blank lines to preserve paragraph separations
    return "\n\n".join([b.text for b in blocks if b.text.strip()])


def _page_span(blocks: List[Bx]) -> Tuple[int, int]:
    if not blocks:
        return (0, 0)
    pgs = [b.page for b in blocks]
    return (min(pgs), max(pgs))


def _first_order(blocks: List[Bx]) -> Tuple[int, int]:
    if not blocks:
        return (0, 0)
    b0 = min(blocks, key=lambda b: (b.page, b.order))
    return (b0.page, b0.order)


def _write_chunks(run_dir: Path, chunks: List[Dict[str, Any]], meta: Dict[str, Any]) -> None:
    cpath = run_dir / "chunks.jsonl"
    mpath = run_dir / "chunks_meta.json"
    with cpath.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    mpath.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[role_chunker] wrote {cpath} ({len(chunks)} chunks)")
    print(f"[role_chunker] wrote {mpath}")


# --------- Strategy: Assessment (questions) ---------

_Q_MAIN = re.compile(r"(?i)^\s*(?:question|q)\s*(\d+)\b")
_Q_NUM = re.compile(r"(?i)^\s*(\d+)\)")
_Q_SUB = re.compile(r"(?i)^\s*([a-d])\)")
_Q_PART = re.compile(r"(?i)^\s*part\s*([a-d])\b")


def _norm_qid(main: Optional[str], sub: Optional[str]) -> Optional[str]:
    if main and sub:
        return f"Q{main.lower()}{sub.lower()}"
    if main:
        return f"Q{main.lower()}"
    return None


def _extract_assessment_questions(blocks: List[Bx]) -> List[Tuple[str, List[Bx]]]:
    """
    Return list of (question_id, blocks_for_question).
    Rule: A question header starts a span until the next header.
    Sub-questions get Qn[a] ids, but live independently.
    """
    # Pass 1: identify headers
    headers: List[Tuple[int, int, str, Optional[str], Optional[str]]] = []
    # tuple: (page, order, header_text, main_num, sub_letter)
    for b in blocks:
        t = b.text.strip()
        main = None
        sub = None
        m = _Q_MAIN.match(t) or _Q_NUM.match(t)
        if m:
            main = m.group(1)
            sub = None
        else:
            ms = _Q_SUB.match(t) or _Q_PART.match(t)
            if ms:
                sub = ms.group(1)
        if main or sub:
            headers.append((b.page, b.order, t, main, sub))

    # Build question spans
    spans: List[Tuple[str, List[Bx]]] = []
    if not headers:
        # No explicit headers → one span (all blocks), no question_id
        return [(None, blocks)]

    # Insert sentinel at end
    sorted_blocks = blocks
    for i, (pg, ord_, hdr, main, sub) in enumerate(headers):
        qid = _norm_qid(main, sub)
        # find start index
        start_idx = next((k for k, tb in enumerate(sorted_blocks)
                          if tb.page == pg and tb.order >= ord_), None)
        if start_idx is None:
            continue
        # end index = next header start
        if i + 1 < len(headers):
            next_pg, next_ord, *_ = headers[i + 1]
            end_idx = next((k for k, tb in enumerate(sorted_blocks)
                            if (tb.page > next_pg) or (tb.page == next_pg and tb.order >= next_ord)), len(sorted_blocks))
        else:
            end_idx = len(sorted_blocks)
        span_blocks = [tb for tb in sorted_blocks[start_idx:end_idx] if tb.text.strip()]
        spans.append((qid, span_blocks))
    return spans


# --------- Strategy: User (answers) ---------

def _attach_question_ids_to_user(blocks: List[Bx], assess_anchor_spans: List[Tuple[str, List[Bx]]]) -> List[Tuple[str, List[Bx]]]:
    """
    Use assessment anchors to tag user blocks to the nearest plausible question.
    Fallback: infer main/sub headers from user text itself.
    """
    # Map assessment headers by page positions for "nearest previous header" heuristic
    assess_headers: List[Tuple[int, int, Optional[str]]] = []
    for qid, blist in assess_anchor_spans:
        if not blist:
            continue
        pg, ord_ = _first_order(blist)
        assess_headers.append((pg, ord_, qid))
    assess_headers.sort()

    # Sweep user blocks and assign qid by nearest prior assessment header on or before page
    user_spans: Dict[Optional[str], List[Bx]] = {}
    for b in blocks:
        # first try local header inference
        t = b.text.strip()
        qid_local = None
        m = _Q_MAIN.match(t) or _Q_NUM.match(t)
        if m:
            qid_local = _norm_qid(m.group(1), None)
        else:
            ms = _Q_SUB.match(t) or _Q_PART.match(t)
            if ms:
                # try to find last main from assessment prior to this block
                last_main = None
                for apg, aord, aqid in reversed(assess_headers):
                    if (apg < b.page) or (apg == b.page and aord <= b.order):
                        last_main = aqid  # could be Qn or Qna
                        break
                if last_main and last_main.startswith("Q") and last_main[1:].isdigit():
                    qid_local = f"{last_main}{ms.group(1).lower()}"
                elif last_main and len(last_main) >= 2 and last_main[1].isdigit():
                    # if already Qna, collapse to that 'Qn' + new sub
                    qnum = "".join([ch for ch in last_main[1:] if ch.isdigit()])
                    qid_local = f"Q{qnum}{ms.group(1).lower()}"

        if not qid_local:
            # nearest assessment header
            cand = None
            for apg, aord, aqid in reversed(assess_headers):
                if (apg < b.page) or (apg == b.page and aord <= b.order):
                    cand = aqid
                    break
            qid_local = cand

        user_spans.setdefault(qid_local, []).append(b)

    # Convert to ordered spans: group contiguous blocks with same qid
    result: List[Tuple[str, List[Bx]]] = []
    last_qid = None
    buf: List[Bx] = []
    for b in blocks:
        qid = None
        # find which bucket this block belongs to by identity
        for q, bl in user_spans.items():
            if b in bl:
                qid = q
                break
        if qid != last_qid and buf:
            result.append((last_qid, buf))
            buf = []
        buf.append(b)
        last_qid = qid
    if buf:
        result.append((last_qid, buf))
    return result


# --------- Strategy: Rubric (rows) ---------

def _cluster_rows_by_y(blocks: List[Bx]) -> List[List[Bx]]:
    """
    Simple y-overlap clustering per page: each cluster approximates one table row.
    """
    rows: List[List[Bx]] = []
    if not blocks:
        return rows
    # group by page first
    by_page: Dict[int, List[Bx]] = {}
    for b in blocks:
        by_page.setdefault(b.page, []).append(b)
    for pg, bl in sorted(by_page.items()):
        bl.sort(key=lambda b: (b.y0, b.x0))
        current: List[Bx] = []
        last_y1: Optional[float] = None
        for b in bl:
            if last_y1 is None:
                current = [b]
                last_y1 = b.y1
                continue
            # start new row when gap is big (no y overlap)
            if b.y0 >= last_y1:
                if current:
                    rows.append(current)
                current = [b]
                last_y1 = b.y1
            else:
                current.append(b)
                last_y1 = max(last_y1, b.y1)
        if current:
            rows.append(current)
    return rows


# --------- Strategy: Assignment (sections) ---------

def _is_header_candidate(b: Bx, fs_thresh: float) -> bool:
    # Prefer font-size based header if available; else fallback to heuristics
    if b.font_size and b.font_size >= fs_thresh:
        return True
    t = b.text.strip()
    if not t:
        return False
    # All-caps short lines or Title-like headings without trailing period
    if (len(t) <= 80 and not t.endswith(".")
        and (t.isupper() or re.match(r"^[A-Z][^\n]{2,}$", t))):
        return True
    return False


# --------- Orchestrators per role ---------

def build_assessment(blocks: List[Bx], max_chars: int, overlap: int) -> List[Dict[str, Any]]:
    spans = _extract_assessment_questions(blocks)
    chunks: List[Dict[str, Any]] = []
    for qid, blist in spans:
        if not blist:
            continue
        full = _join_blocks(blist)
        # Prefer atomic question chunks; split only if extremely long
        pieces = _window_text(full, max_chars=max_chars * 2, overlap=overlap)  # effectively one piece
        for j, (s0, s1, text) in enumerate(pieces, 1):
            pid = f"{blist[0].page:04d}:{blist[0].order:06d}"
            chunks.append({
                "id": f"{pid}-assess-{j}",
                "role": "assessment",
                "question_id": qid,
                "criterion": None,
                "section": None,
                "page_span": list(_page_span(blist)),
                "block_ids": [b.bid for b in blist],
                "order": list(_first_order(blist)),
                "text": text.strip(),
            })
    return chunks


def build_user(blocks: List[Bx], assess_spans: List[Tuple[str, List[Bx]]],
               max_chars: int, overlap: int, min_par_len: int) -> List[Dict[str, Any]]:
    buckets = _attach_question_ids_to_user(blocks, assess_spans)
    chunks: List[Dict[str, Any]] = []
    for qid, blist in buckets:
        if not blist:
            continue
        # rolling inside the answer span
        text = _join_blocks(blist)
        if not text.strip():
            continue
        pieces = _window_text(text, max_chars=max_chars, overlap=overlap)
        for j, (s0, s1, tx) in enumerate(pieces, 1):
            if len(tx.strip()) < min_par_len:
                continue
            pid = f"{blist[0].page:04d}:{blist[0].order:06d}"
            chunks.append({
                "id": f"{pid}-user-{j}",
                "role": "user",
                "question_id": qid,
                "criterion": None,
                "section": None,
                "page_span": list(_page_span(blist)),
                "block_ids": [b.bid for b in blist],
                "order": list(_first_order(blist)),
                "text": tx.strip(),
            })
    return chunks


def build_rubric(blocks: List[Bx]) -> List[Dict[str, Any]]:
    rows = _cluster_rows_by_y(blocks)
    chunks: List[Dict[str, Any]] = []
    for ridx, row in enumerate(rows, 1):
        if not row:
            continue
        # leftmost block approximates criterion
        left = min(row, key=lambda b: (b.x0, b.y0))
        criterion = left.text.strip().splitlines()[0][:120] if left.text.strip() else f"Row {ridx}"
        # aggregate row text in left-to-right order
        row_sorted = sorted(row, key=lambda b: (b.x0, b.y0))
        text = " | ".join([b.text.strip() for b in row_sorted if b.text.strip()])
        pid = f"{row_sorted[0].page:04d}:{row_sorted[0].order:06d}"
        chunks.append({
            "id": f"{pid}-rubric-{ridx}",
            "role": "rubric",
            "question_id": None,
            "criterion": criterion,
            "section": None,
            "page_span": list(_page_span(row_sorted)),
            "block_ids": [b.bid for b in row_sorted],
            "order": list(_first_order(row_sorted)),
            "text": text,
        })
    return chunks


def build_assignment(blocks: List[Bx], max_chars: int, overlap: int, min_par_len: int) -> List[Dict[str, Any]]:
    # Determine font-size threshold (top quartile) for headers if available
    fs_vals = [b.font_size for b in blocks if b.font_size > 0]
    fs_thresh = 0.0
    if fs_vals:
        fs_vals_sorted = sorted(fs_vals)
        fs_thresh = fs_vals_sorted[int(0.75 * (len(fs_vals_sorted) - 1))]
    # Identify headers
    headers: List[Tuple[int, int, str]] = []
    for b in blocks:
        if _is_header_candidate(b, fs_thresh):
            headers.append((b.page, b.order, b.text.strip()))
    headers = sorted(set(headers))  # dedupe
    # Build sections
    chunks: List[Dict[str, Any]] = []
    if not headers:
        # no headers → one big rolling segment over all blocks
        text = _join_blocks(blocks)
        for j, (s0, s1, tx) in enumerate(_window_text(text, max_chars=max_chars, overlap=overlap), 1):
            if len(tx.strip()) < min_par_len:
                continue
            pid = f"{blocks[0].page:04d}:{blocks[0].order:06d}"
            chunks.append({
                "id": f"{pid}-assign-{j}",
                "role": "assignment",
                "question_id": None,
                "criterion": None,
                "section": None,
                "page_span": list(_page_span(blocks)),
                "block_ids": [b.bid for b in blocks],
                "order": list(_first_order(blocks)),
                "text": tx.strip(),
            })
        return chunks

    # index into blocks for each header
    for i, (pg, ord_, title) in enumerate(headers):
        # find start
        start_idx = next((k for k, tb in enumerate(blocks) if tb.page == pg and tb.order >= ord_), None)
        if start_idx is None:
            continue
        # end = next header start or end of doc
        if i + 1 < len(headers):
            next_pg, next_ord, _ = headers[i + 1]
            end_idx = next((k for k, tb in enumerate(blocks)
                            if (tb.page > next_pg) or (tb.page == next_pg and tb.order >= next_ord)), len(blocks))
        else:
            end_idx = len(blocks)
        span = [tb for tb in blocks[start_idx:end_idx] if tb.text.strip()]
        if not span:
            continue
        text = _join_blocks(span)
        for j, (s0, s1, tx) in enumerate(_window_text(text, max_chars=max_chars, overlap=overlap), 1):
            if len(tx.strip()) < min_par_len:
                continue
            pid = f"{span[0].page:04d}:{span[0].order:06d}"
            chunks.append({
                "id": f"{pid}-assign-{i+1:02d}-{j}",
                "role": "assignment",
                "question_id": None,
                "criterion": None,
                "section": title[:120],
                "page_span": list(_page_span(span)),
                "block_ids": [b.bid for b in span],
                "order": list(_first_order(span)),
                "text": tx.strip(),
            })
    return chunks


# --------- Main ---------

def main() -> None:
    ap = argparse.ArgumentParser(description="Role-aware blocks chunker")
    ap.add_argument("--run_dir", type=Path, required=True)
    ap.add_argument("--pdf", type=Path, required=True)
    ap.add_argument("--role", type=str, required=True,
                    choices=["user", "assessment", "rubric", "assignment"])
    ap.add_argument("--max_chars", type=int, default=1100)
    ap.add_argument("--overlap", type=int, default=150)
    ap.add_argument("--min_par_len", type=int, default=40)
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    blocks_path = run_dir / "blocks_refined.jsonl"

    # Try blocks first, then fallback to OCR pages
    blocks = _read_blocks(blocks_path)
    source = "blocks_refined.jsonl"
    if not blocks:
        blocks = _read_ocr_pages(run_dir)
        source = "ocr_pages"

    if not blocks:
        print(f"[role_chunker] No blocks or OCR pages found in {run_dir}/outputs/", file=sys.stderr)
        sys.exit(1)

    # Assessment spans may be used to align user answers
    assess_spans: List[Tuple[str, List[Bx]]] = []
    if args.role in ("user",):
        # Try to read assessment blocks alongside (same runs parent)
        assess_dir = run_dir.parent / "assessment_e2e"
        assess_blocks = _read_blocks(assess_dir / "blocks_refined.jsonl")
        if not assess_blocks:
            assess_blocks = _read_ocr_pages(assess_dir)
        assess_spans = _extract_assessment_questions(assess_blocks)

    if args.role == "assessment":
        chunks = build_assessment(blocks, max_chars=args.max_chars, overlap=args.overlap)
    elif args.role == "user":
        chunks = build_user(blocks, assess_spans, max_chars=args.max_chars,
                            overlap=args.overlap, min_par_len=args.min_par_len)
    elif args.role == "rubric":
        chunks = build_rubric(blocks)
    elif args.role == "assignment":
        chunks = build_assignment(blocks, max_chars=args.max_chars,
                                  overlap=args.overlap, min_par_len=args.min_par_len)
    else:
        print(f"[role_chunker] Unknown role: {args.role}", file=sys.stderr)
        sys.exit(2)

    meta = {
        "run_dir": str(run_dir),
        "pdf": str(args.pdf),
        "role": args.role,
        "source": source,
        "params": {
            "max_chars": args.max_chars,
            "overlap": args.overlap,
            "min_par_len": args.min_par_len,
        },
        "chunks": len(chunks),
    }
    _write_chunks(run_dir, chunks, meta)


if __name__ == "__main__":
    main()
