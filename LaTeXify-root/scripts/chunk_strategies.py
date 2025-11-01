#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chunk strategies for LaTeXify

Modes
-----
- user        → hybrid structural-by-question + rolling
- assessment  → one chunk per question/sub-question (roll if oversized)
- rubric      → one chunk per rubric row/criterion
- assignment  → section headers → rolling within section

Resilience
----------
- Works even when blocks_refined.jsonl is missing by synthesizing blocks from
  outputs/<backend>/page-*.md (simple heuristics).
- Strategies now do a 2-pass attempt:
    pass-1: use requested min_par_len
    pass-2: if empty, relax thresholds and broaden detectors (min_par_len=1)

Chunk Schema (each JSONL row)
-----------------------------
{
  "id": "q1a-chunk-p3-00",
  "text": "...",
  "page": 3,
  "bbox": [x0, y0, x1, y1] | null,
  "block_type": "Text" | "Header" | "Formula" | "Table" | "TableRow" | null,
  "source_backend": "qwen2-vl-ocr-2b" | null,
  "semantic_id": "Q1a" | "Instructions" | "Use of Appropriate Symbols" | "misc",
  "flags": {"high_ocr_disagreement": false, "low_confidence": false}
}
"""
from __future__ import annotations

import re
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# =========================
# Loading / fallback blocks
# =========================

def _load_jsonl(path: Path) -> List[dict]:
    out: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                # tolerate comments or junk lines
                continue
    return out


def load_blocks(run_dir: str, prefer_backends: Optional[List[str]] = None) -> List[dict]:
    """
    Prefer structured blocks_refined.jsonl if present, else synthesize from
    outputs/<backend>/page-*.md. This function only *loads*; chunking is done
    downstream.

    Expected refined block fields (best-effort; optional):
      page, bbox, block_type/label, text, row_id, source_backend,
      disagreement, low_conf
    """
    run_path = Path(run_dir)
    refined = run_path / "blocks_refined.jsonl"
    if refined.exists():
        blocks = _load_jsonl(refined)
        for b in blocks:
            b.setdefault("block_type", b.get("label"))
            b.setdefault("source_backend", None)
        return blocks

    # Fallback: synthesize from outputs/<backend>/page-*.md
    outputs_glob = sorted((run_path / "outputs").glob("*"))
    if prefer_backends:
        order = {name: i for i, name in enumerate(prefer_backends)}
        outputs_glob = sorted(outputs_glob, key=lambda p: order.get(p.name, 10_000))

    for backend in outputs_glob:
        page_md = sorted(list(backend.glob("page-*.md"))) or sorted(list(backend.glob("**/page-*.md")))
        if not page_md:
            continue

        synth: List[dict] = []
        for p in page_md:
            page_no = _num_from_filename(p.name)
            text = p.read_text(encoding="utf-8", errors="ignore")
            for para in _split_paragraphs(text):
                if not para.strip():
                    continue
                synth.append({
                    "page": page_no,
                    "bbox": None,  # unknown
                    "block_type": _guess_block_type(para),
                    "text": para.strip(),
                    "label": None,
                    "row_id": None,
                    "source_backend": backend.name,
                    "disagreement": None,
                    "low_conf": False,
                })
        if synth:
            return synth

    return []


def _num_from_filename(name: str) -> Optional[int]:
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else None


def _split_paragraphs(text: str) -> List[str]:
    # robust split on 1+ blank lines; also split very long paras by sentence windows
    raw = [c.strip() for c in re.split(r"\n\s*\n", text) if c.strip()]
    out: List[str] = []
    for para in raw:
        if len(para) <= 1400:
            out.append(para)
            continue
        # break into ~600–900 char windows on sentence boundaries
        chunks = []
        buf = []
        total = 0
        for sent in re.split(r"(?<=[\.\?\!])\s+", para):
            if total + len(sent) > 900 and buf:
                chunks.append(" ".join(buf))
                buf, total = [sent], len(sent)
            else:
                buf.append(sent)
                total += len(sent)
        if buf:
            chunks.append(" ".join(buf))
        out.extend(chunks)
    return out


def _guess_block_type(para: str) -> str:
    if re.search(r"^\s*(\$\$|\\\[)", para, re.M) or re.search(r"\$(.+?)\$", para):
        return "Formula"
    if any(c in para for c in ["|", "─", "┼", "│"]) and len(para) > 30:
        return "Table"
    if re.match(r"^\s*(Question|Q\s*\d+|Part\s+[IVX]+|[A-Z][A-Za-z0-9\s\-]{2,40}:)\b", para, re.I):
        return "Header"
    return "Text"


# =========
# Helpers
# =========

_Q_MAIN_ANYWHERE = re.compile(r"(?:^|\b)(?:Q(?:uestion)?\s*)?(\d+)(?=[\)\.\:\-–\s])", re.I)
_Q_ENUM_LINE     = re.compile(r"^\s*(\d+)\s*[\)\.\-–]\s+", re.I | re.M)
_Q_SUB_LINE      = re.compile(r"^\s*([a-z])\s*[\)\.\-–]\s+", re.I | re.M)
_Q_SUB_WORD      = re.compile(r"(?:^|\b)(?:part|section)\s*([a-z])\b", re.I)
_Q_COMPACT       = re.compile(r"\bQ(\d+)([a-z])\b", re.I)

def parse_qid_relaxed(text: str, last_main: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Best-effort parser for question anchors.
    Returns (main, sub) where main like 'Q3' and sub like 'a'.
    """
    s = text or ""

    # Exact compact form "Q3a"
    m = _Q_COMPACT.search(s)
    if m:
        return f"Q{m.group(1)}", m.group(2).lower()

    # Line enumerations at start
    m = _Q_ENUM_LINE.search(s)
    if m:
        main = f"Q{m.group(1)}"
        # In case it’s like "1) a) ..." we also try sub on same text
        sub = None
        ms = _Q_SUB_LINE.search(s)
        if ms:
            sub = ms.group(1).lower()
        return main, sub

    # Anywhere "Question 3", "Q 3", "Q3"
    m = _Q_MAIN_ANYWHERE.search(s)
    main = f"Q{m.group(1)}" if m else last_main

    # Sub like "a) " or "Part a"
    ms = _Q_SUB_LINE.search(s) or _Q_SUB_WORD.search(s)
    sub = ms.group(1).lower() if ms else None

    if not main and sub:
        # seen a sub without main; treat as misc sub
        return None, sub
    return main, sub


def y_mid(bbox: Optional[List[float]]) -> Optional[float]:
    if not bbox or len(bbox) != 4:
        return None
    return 0.5 * (bbox[1] + bbox[3])


def _tail_text(text: str, overlap: int) -> str:
    if not text or overlap <= 0:
        return ""
    if overlap >= len(text):
        return text
    start = max(0, len(text) - overlap)
    # try to avoid cutting a word
    while start > 0 and text[start - 1].isalnum():
        start += 1
        if start >= len(text):
            return ""
    return text[start:]


def _span_bbox(bboxes: List[Optional[List[float]]]) -> Optional[List[float]]:
    xs: List[float] = []
    ys: List[float] = []
    for b in bboxes:
        if not b or len(b) != 4:
            continue
        x0, y0, x1, y1 = b
        xs.extend([x0, x1]); ys.extend([y0, y1])
    if not xs or not ys:
        return None
    return [min(xs), min(ys), max(xs), max(ys)]


def _majority(items: List[Optional[str]]) -> Optional[str]:
    counts: Dict[str, int] = {}
    for it in items:
        if not it:
            continue
        counts[it] = counts.get(it, 0) + 1
    if not counts:
        return None
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def _merge_flags(blocks: List[dict]) -> dict:
    flags = {
        "high_ocr_disagreement": False,
        "low_confidence": False,
    }
    for b in blocks:
        if b.get("disagreement") is not None:
            try:
                if float(b["disagreement"]) >= 0.25:
                    flags["high_ocr_disagreement"] = True
            except Exception:
                pass
        if b.get("low_conf") is True:
            flags["low_confidence"] = True
    return flags


def _collect_metadata(blocks: List[dict],
                      page: Optional[int],
                      bbox: Optional[List[float]],
                      block_type: Optional[str],
                      label: Optional[str],
                      source_backend: Optional[str],
                      semantic_id: Optional[str],
                      flags: dict) -> Dict[str, Any]:
    pages = [b.get("page") for b in blocks if b.get("page") is not None]
    page_values: List[int] = []
    for p in pages:
        try:
            page_values.append(int(p))
        except Exception:
            continue
    unique_pages = sorted(set(page_values))
    block_ids = [
        str(b.get("block_id") or b.get("id") or b.get("row_id"))
        for b in blocks if b.get("block_id") or b.get("id") or b.get("row_id")
    ]
    backend_candidates = {
        str(b.get("source_backend") or b.get("backend") or b.get("ocr_backend"))
        for b in blocks if b.get("source_backend") or b.get("backend") or b.get("ocr_backend")
    }
    label_candidates = {
        str(b.get("label") or b.get("block_type"))
        for b in blocks if b.get("label") or b.get("block_type")
    }
    meta: Dict[str, Any] = {
        "page": page,
        "pages": unique_pages or None,
        "page_span": [unique_pages[0], unique_pages[-1]] if unique_pages else None,
        "bbox": bbox,
        "block_type": block_type,
        "label": label,
        "semantic_id": semantic_id,
        "block_ids": block_ids or None,
        "source_backend": source_backend,
        "source_backends": sorted(backend_candidates) or None,
        "labels": sorted(label_candidates) or None,
        "flags": flags if any(flags.values()) else None,
    }
    return {k: v for k, v in meta.items() if v not in (None, [], {})}


def _make_chunk_id(semantic_id: Optional[str], page: Optional[int], idx: int) -> str:
    sid = (semantic_id or "auto").lower().replace(" ", "-")
    p = f"p{page}" if page is not None else "px"
    return f"{sid}-chunk-{p}-{idx:02d}"


# =========================
# Rolling pack (2-pass)
# =========================

def rolling_pack_blocks(blocks: List[dict],
                        max_chars: int = 1000,
                        overlap: int = 150,
                        min_par_len: int = 40,
                        semantic_id: Optional[str] = None) -> List[dict]:
    """
    Concatenate adjacent blocks until threshold; overlap tail for context.
    Two-pass: if nothing after filtering, relax to min_par_len=1.
    """
    def _pack(filtered: List[dict]) -> List[dict]:
        chunks: List[dict] = []
        buf: List[dict] = []
        total = 0

        def flush():
            nonlocal buf, total
            if not buf:
                return
            text = "\n\n".join(b["text"].strip() for b in buf if b.get("text"))
            if not text:
                buf, total = [], 0
                return
            pages = [b.get("page") for b in buf if b.get("page") is not None]
            page = pages[0] if pages else None
            bbox = _span_bbox([b.get("bbox") for b in buf])
            block_type = _majority([b.get("block_type") for b in buf])
            source_backend = _majority([b.get("source_backend") for b in buf])
            label = _majority([b.get("label") or b.get("block_type") for b in buf])
            flags = _merge_flags(buf)
            chunk_id = _make_chunk_id(semantic_id, page, len(chunks))
            chunks.append({
                "id": chunk_id,
                "text": text,
                "page": page,
                "bbox": bbox,
                "block_type": block_type,
                "label": label,
                "source_backend": source_backend,
                "semantic_id": semantic_id,
                "flags": flags,
                "metadata": _collect_metadata(buf, page, bbox, block_type, label, source_backend, semantic_id, flags),
            })
            tail = _tail_text(text, overlap)
            buf, total = [], 0
            if tail:
                buf.append({
                    "text": tail,
                    "page": page,
                    "bbox": bbox,
                    "block_type": block_type,
                    "source_backend": source_backend,
                    "flags": {}
                })
                total = len(tail)

        for b in filtered:
            t = (b.get("text") or "").strip()
            if not t:
                continue
            if total + len(t) + 2 > max_chars and buf:
                flush()
            buf.append(b)
            total += len(t) + 2
        flush()
        return chunks

    pre = [b for b in blocks if isinstance(b.get("text"), str) and len(b["text"].strip()) >= min_par_len]
    chunks = _pack(pre)
    if chunks:
        return chunks
    # Relax
    return _pack([b for b in blocks if isinstance(b.get("text"), str) and len(b["text"].strip()) >= 1])


# =========================
# Strategy implementations
# =========================

def _order_key(b: dict) -> Tuple[int, float]:
    return (b.get("page") or 10**9, y_mid(b.get("bbox") or [0, 0, 0, 0]) or 0.0)


def chunks_for_user(blocks: List[dict], max_chars: int, overlap: int, min_par_len: int) -> List[dict]:
    """
    Group by detected QID (Q3, Q3a...). If no anchors found, create a stable
    rolling pack per page group so we still get multiple chunks.
    """
    ordered = sorted(blocks, key=_order_key)
    groups: Dict[str, List[dict]] = {}
    last_main: Optional[str] = None
    current_sid: Optional[str] = None

    for b in ordered:
        m, sub = parse_qid_relaxed(b.get("text", ""), last_main)
        if m:
            last_main = m
        sid = f"{m}{sub}" if m and sub else (m or current_sid or "misc")
        if m or sub:
            current_sid = sid
        groups.setdefault(sid, []).append(b)

    # If everything collapsed to "misc", split by page into pseudo question groups.
    if list(groups.keys()) == ["misc"]:
        page_groups: Dict[str, List[dict]] = {}
        for b in ordered:
            pg = b.get("page")
            key = f"misc-p{pg}" if pg is not None else "misc"
            page_groups.setdefault(key, []).append(b)
        groups = page_groups

    chunks: List[dict] = []
    for sid, blist in groups.items():
        chunks.extend(rolling_pack_blocks(blist, max_chars, overlap, min_par_len, semantic_id=sid))
    return chunks


def chunks_for_assessment(blocks: List[dict], max_chars: int, overlap: int, min_par_len: int) -> List[dict]:
    """
    One chunk per question/sub-question; if a group is too big, rolling-pack inside it.
    If no anchors are found (scanned assessments sometimes lose numbering), create
    page-based groups with small rolling windows.
    """
    ordered = sorted(blocks, key=_order_key)
    groups: List[Tuple[str, List[dict]]] = []
    last_main: Optional[str] = None
    current_sid: Optional[str] = None
    current: List[dict] = []

    def flush():
        nonlocal current, current_sid
        if not current:
            return
        groups.append((current_sid or "misc", current))
        current, current_sid = [], None

    for b in ordered:
        text = b.get("text", "") or ""
        m, sub = parse_qid_relaxed(text, last_main)
        if m:
            last_main = m
        sid = f"{m}{sub}" if m and sub else m
        if sid:
            flush()
            current_sid = sid
            current = [b]
        else:
            if current_sid is None:
                current_sid = "misc"
            current.append(b)
    flush()

    if groups and not all(k == "misc" for k, _ in groups):
        # Normal path
        chunks: List[dict] = []
        for sid, blist in groups:
            if sid == "misc":
                continue
            tot_len = len("\n\n".join((b.get("text") or "").strip() for b in blist))
            if tot_len <= max_chars:
                chunks.extend(rolling_pack_blocks(blist, 10**9, 0, 1, semantic_id=sid))
            else:
                chunks.extend(rolling_pack_blocks(blist, max_chars, overlap, min_par_len, semantic_id=sid))
        # dump a small 'misc' section if it has content
        for sid, blist in groups:
            if sid != "misc":
                continue
            if any((b.get("text") or "").strip() for b in blist):
                chunks.extend(rolling_pack_blocks(blist, max_chars, overlap, min_par_len, semantic_id="misc"))
        return chunks

    # Fallback grouping by page windows to ensure more than one chunk
    page_groups: Dict[int, List[dict]] = {}
    for b in ordered:
        page_groups.setdefault(b.get("page") or -1, []).append(b)

    chunks: List[dict] = []
    for pg, blist in sorted(page_groups.items()):
        sid = f"p{pg}"
        # smaller windows to force multiple chunks
        chunks.extend(rolling_pack_blocks(blist, max_chars=max_chars//2, overlap=overlap//2, min_par_len=min_par_len//2 or 1, semantic_id=sid))
    return chunks


def chunks_for_rubric(blocks: List[dict], max_chars: int, overlap: int, min_par_len: int) -> List[dict]:
    """
    Prefer explicit row_id; else bucket by y-midline bands per page.
    Name criterion from header-like text or shortest meaningful phrase.
    """
    rows: Dict[str, List[dict]] = {}
    explicit = [b for b in blocks if b.get("row_id")]
    if explicit:
        for b in explicit:
            rows.setdefault(str(b["row_id"]), []).append(b)
    else:
        ordered = sorted(blocks, key=_order_key)
        for b in ordered:
            ym = y_mid(b.get("bbox")) or 0.0
            band = f"p{b.get('page','x')}-y{int(round(ym/20.0)*20)}"
            rows.setdefault(band, []).append(b)

    chunks: List[dict] = []
    for rid, blist in rows.items():
        blist = sorted(blist, key=_order_key)
        text_parts = [b.get("text","").strip() for b in blist if b.get("text")]
        row_text = " ".join(t for t in text_parts if t)
        criterion = _extract_rubric_criterion(text_parts) or f"row-{rid}"
        packed = rolling_pack_blocks(
            [{"text": row_text, "page": blist[0].get("page"),
              "bbox": _span_bbox([b.get("bbox") for b in blist]),
              "block_type": "TableRow",
              "source_backend": _majority([b.get("source_backend") for b in blist]),
              "flags": _merge_flags(blist)}],
            max_chars=max_chars, overlap=overlap, min_par_len=1, semantic_id=criterion
        )
        chunks.extend(p for p in packed if p.get("text"))
    return chunks


def _extract_rubric_criterion(row_texts: List[str]) -> Optional[str]:
    for t in row_texts:
        head = t.strip().split(".")[0]
        if 3 <= len(head) <= 64 and _is_header_text(head):
            return head
    cand = sorted((s for s in row_texts if len(s.strip()) >= 6), key=len)
    return cand[0].strip() if cand else None


def _is_header_text(txt: str) -> bool:
    line = (txt or "").strip().splitlines()[0] if txt else ""
    if len(line) < 3:
        return False
    if re.match(r"^\s*(important|instructions|submission|format|policy|overview|ai\s+use|academic|evaluation|grading|rubric)\b", line, re.I):
        return True
    if len(line) <= 72 and (line.isupper() or re.match(r"^[A-Z][A-Za-z0-9 \-,:/]{2,}$", line)):
        return True
    return False


def chunks_for_assignment(blocks: List[dict], max_chars: int, overlap: int, min_par_len: int) -> List[dict]:
    """
    Detect sections (Instructions, Submission, Formatting, Policy, etc.)
    and roll inside each. If nothing is detected, ensure page-based
    multiple chunks.
    """
    ordered = sorted(blocks, key=_order_key)
    sections: List[Tuple[str, List[dict]]] = []
    current_name = "Intro"
    current: List[dict] = []

    def flush():
        nonlocal current, current_name
        if not current:
            return []
        out = rolling_pack_blocks(current, max_chars, overlap, min_par_len, semantic_id=current_name)
        current = []
        return out

    chunks: List[dict] = []
    header_seen = False
    for b in ordered:
        txt = b.get("text", "") or ""
        if _is_header_text(txt):
            header_seen = True
            chunks.extend(flush())
            current_name = txt.strip().splitlines()[0][:80]
        current.append(b)
    chunks.extend(flush())

    if chunks:
        return chunks

    # Fallback: enforce multi-chunk by smaller windows per page
    page_groups: Dict[int, List[dict]] = {}
    for b in ordered:
        page_groups.setdefault(b.get("page") or -1, []).append(b)

    for pg, blist in sorted(page_groups.items()):
        chunks.extend(rolling_pack_blocks(blist,
                                          max_chars=max_chars//2,
                                          overlap=overlap//2,
                                          min_par_len=min_par_len//2 or 1,
                                          semantic_id=f"p{pg}"))
    return chunks


# =========================
# Public selector
# =========================

def build_chunks_for_mode(blocks: List[dict],
                          mode: str,
                          max_chars: int,
                          overlap: int,
                          min_par_len: int) -> List[dict]:
    mode = mode.lower()
    if mode == "user":
        return chunks_for_user(blocks, max_chars, overlap, min_par_len)
    if mode == "assessment":
        return chunks_for_assessment(blocks, max_chars, overlap, min_par_len)
    if mode == "rubric":
        return chunks_for_rubric(blocks, max_chars, overlap, min_par_len)
    if mode == "assignment":
        return chunks_for_assignment(blocks, max_chars, overlap, min_par_len)
    raise ValueError(f"unknown mode: {mode}")
