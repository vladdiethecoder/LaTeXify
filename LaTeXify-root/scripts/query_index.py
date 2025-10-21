# scripts/query_index.py
from __future__ import annotations
import argparse, json, math
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import faiss  # type: ignore
from sentence_transformers import SentenceTransformer

def _load_index(run_dir: Path):
    index = faiss.read_index(str(run_dir / "faiss.index"))
    meta = json.loads((run_dir / "faiss.meta.json").read_text(encoding="utf-8"))
    return index, meta

def _parse_pages(pages_arg: Optional[str]) -> Optional[Set[int]]:
    if not pages_arg:
        return None
    out: Set[int] = set()
    for part in pages_arg.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            try:
                start = int(a); end = int(b)
            except ValueError:
                continue
            if start > end:
                start, end = end, start
            out.update(range(start, end + 1))
        else:
            try:
                out.add(int(part))
            except ValueError:
                pass
    return out or None

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--pages", type=str, default=None, help="e.g., '3-4,7'")
    p.add_argument("query", type=str)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    index, meta = _load_index(run_dir)
    encoder_name = meta["encoder"]
    chunks_meta: List[Dict] = meta["chunks_meta"]
    model = SentenceTransformer(encoder_name)

    # Encode query
    qv = model.encode([args.query], normalize_embeddings=True, convert_to_numpy=True)
    # Search more than k if we need to post-filter by pages
    want_pages = _parse_pages(args.pages)
    overshoot = max(args.k * 10, args.k + 10)
    D, I = index.search(qv, min(overshoot, len(chunks_meta)))

    # Gather results with optional page filtering
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        meta_i = chunks_meta[idx]
        if want_pages and meta_i["page_num"] not in want_pages:
            continue
        results.append((float(score), idx, meta_i))
        if len(results) >= args.k:
            break

    # Pretty print results
    print(f"\nTop {len(results)} results for: {args.query!r}\n")
    for rank, (score, idx, cm) in enumerate(results, start=1):
        # Load the chunk text from chunks.jsonl for an excerpt
        # (lazy load to avoid big memory; optional optimization)
        # Here we just show a truncated view from source for simplicity:
        # If you want exact text, open chunks.jsonl and line-scan for chunk_id.
        preview = f"{cm['source_image']} | page {cm['page_num']} | model={cm['model']}"
        print(f"{rank:02d}. score={score:.4f}  {preview}")

if __name__ == "__main__":
    main()
