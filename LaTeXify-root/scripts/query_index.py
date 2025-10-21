# scripts/query_index.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def _load_index(run_dir: Path):
    index = faiss.read_index(str(run_dir / "faiss.index"))
    meta = json.loads((run_dir / "faiss.meta.json").read_text(encoding="utf-8"))
    return index, meta

def _encode(q: str, model_name: str, device: Optional[str]) -> np.ndarray:
    model = SentenceTransformer(model_name, device=device or "cpu")
    q_emb = model.encode([q], convert_to_numpy=True, normalize_embeddings=True)
    return q_emb

def _page_filter(meta: dict, pages: Optional[List[int]]) -> Optional[np.ndarray]:
    if not pages:
        return None
    # build mask of ids to keep
    id_to_ref = meta["id_to_ref"]
    keep = np.zeros(len(id_to_ref), dtype=bool)
    page_set = set(pages)
    for i, ref in enumerate(id_to_ref):
        p = ref.get("page_idx")
        if p in page_set:
            keep[i] = True
    return keep

def _parse_pages(spec: Optional[str]) -> Optional[List[int]]:
    if not spec:
        return None
    out: List[int] = []
    for token in spec.split(","):
        token = token.strip()
        if "-" in token:
            a, b = token.split("-", 1)
            a, b = int(a), int(b)
            out.extend(range(min(a, b), max(a, b) + 1))
        elif token:
            out.append(int(token))
    return sorted(set(out))

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, type=str)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--pages", type=str, default=None, help="e.g., '3-4,7'")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("query", type=str)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    index, meta = _load_index(run_dir)
    model_name = meta["encoder"]

    q_emb = _encode(args.query, model_name=model_name, device=args.device)

    # Optional page filter by brute-force masking
    mask = _page_filter(meta, _parse_pages(args.pages))
    if mask is not None:
        # Reconstruct a masked view using an IDMap (copy subset)
        # Simpler: search on full index then post-filter
        D, I = index.search(q_emb, max(args.k * 10, args.k))
        hits = []
        for dist, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            if mask[idx]:
                hits.append((idx, float(dist)))
            if len(hits) >= args.k:
                break
    else:
        D, I = index.search(q_emb, args.k)
        hits = [(int(i), float(d)) for i, d in zip(I[0], D[0]) if i >= 0]

    # Print friendly results
    id_to_ref = meta["id_to_ref"]
    for rank, (idx, score) in enumerate(hits, start=1):
        ref = id_to_ref[idx]
        page_label = f"p{ref.get('page_idx')}" if ref.get("page_idx") else ref.get("page")
        print(f"[{rank}] score={score:.4f}  id={ref['id']}  page={page_label}  span={ref['char_start']}-{ref['char_end']}")

if __name__ == "__main__":
    main()
