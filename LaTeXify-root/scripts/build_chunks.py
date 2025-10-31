#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_chunks.py

Creates semantically-structured chunks from OCR/layout blocks, per role:
  - user        → hybrid structural-by-question + rolling
  - assessment  → one chunk per question/sub-question
  - rubric      → one chunk per rubric row/criterion
  - assignment  → section headers → rolling

Resilient: if a strategy still yields zero chunks, it retries with relaxed
thresholds and, as a last resort, emits a deterministic rolled chunk.

Outputs:
  <run_dir>/chunks.jsonl
  <run_dir>/chunks_meta.json
"""
from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import List, Optional

from chunk_strategies import (
    load_blocks,
    build_chunks_for_mode,
    rolling_pack_blocks,  # deterministic last-resort packing
)

ROLE_HINTS = {
    "user": "user",
    "assessment": "assessment",
    "rubric": "rubric",
    "assignment": "assignment",
}


def infer_mode(run_dir: str, pdf: str) -> str:
    s = f"{run_dir} {pdf}".lower()
    for k, v in ROLE_HINTS.items():
        if k in s:
            return v
    if "assess" in s or "question" in s:
        return "assessment"
    if "rubric" in s:
        return "rubric"
    if "user" in s:
        return "user"
    return "assignment"


def write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="Per-PDF run dir (e.g., dev/runs/user_e2e)")
    ap.add_argument("--pdf", required=True, help="Source PDF (used for mode inference only)")
    ap.add_argument("--mode", default="auto", choices=["auto", "user", "assessment", "rubric", "assignment"])
    ap.add_argument("--max_chars", type=int, default=1100)
    ap.add_argument("--overlap", type=int, default=150)
    ap.add_argument("--min_par_len", type=int, default=40)
    ap.add_argument("--prefer", type=str, default="", help="comma list of OCR backend names in priority order")
    args = ap.parse_args()

    prefer = [p.strip() for p in args.prefer.split(",") if p.strip()] if args.prefer else None
    mode = infer_mode(args.run_dir, args.pdf) if args.mode == "auto" else args.mode

    blocks = load_blocks(args.run_dir, prefer_backends=prefer)
    if not blocks:
        print(f"[chunker] no blocks available in {args.run_dir}")
        write_jsonl(Path(args.run_dir) / "chunks.jsonl", [])
        (Path(args.run_dir) / "chunks_meta.json").write_text(
            json.dumps({"count": 0, "mode": mode, "max_chars": args.max_chars,
                        "overlap": args.overlap, "min_par_len": args.min_par_len}, indent=2),
            encoding="utf-8"
        )
        return

    # Pass 1: strategy as requested
    chunks = build_chunks_for_mode(blocks, mode=mode,
                                   max_chars=args.max_chars,
                                   overlap=args.overlap,
                                   min_par_len=args.min_par_len)

    # Pass 2: If still empty, retry strategy with relaxed parameters to force multi-chunk
    if not chunks:
        print(f"[chunker] strategy produced 0 chunks (mode={mode}); retrying with relaxed thresholds")
        chunks = build_chunks_for_mode(blocks, mode=mode,
                                       max_chars=max(400, args.max_chars // 2),
                                       overlap=max(40, args.overlap // 2),
                                       min_par_len=1)

    # Final fallback: single rolled chunk (deterministic)
    if not chunks:
        print(f"[chunker] still 0 chunks after relax (mode={mode}); running last-resort pack")
        chunks = rolling_pack_blocks(blocks,
                                     max_chars=args.max_chars,
                                     overlap=args.overlap,
                                     min_par_len=1,
                                     semantic_id="misc")

    write_jsonl(Path(args.run_dir) / "chunks.jsonl", chunks)
    (Path(args.run_dir) / "chunks_meta.json").write_text(
        json.dumps({"count": len(chunks), "mode": mode, "max_chars": args.max_chars,
                    "overlap": args.overlap, "min_par_len": args.min_par_len}, indent=2),
        encoding="utf-8"
    )
    print(f"[chunker] wrote {Path(args.run_dir) / 'chunks.jsonl'} ({len(chunks)} chunks)")
    print(f"[chunker] wrote {Path(args.run_dir) / 'chunks_meta.json'}")


if __name__ == "__main__":
    main()
