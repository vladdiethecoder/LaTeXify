# scripts/query_index.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def _load_index(run_dir: Path):
    index = faiss.read_index(str(run_dir / "faiss.index"))
    meta = json.loads((run_dir / "faiss.meta.json").read_text(encoding="utf-8"))
    return index, meta


def _parse_pages(spec: Optional[str]) -> Optional[List[int]]:
    if not spec:
        return None
    pages: List[int] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            a, b = token.split("-", 1)
            try:
                a, b = int(a), int(b)
                pages.extend(list(range(min(a, b), max(a, b) + 1)))
            except Exception:
                continue
        else:
            try:
                pages.append(int(token))
            except Exception:
                continue
    return sorted(set(pages)) or None


def _encode(q: str, model_name="sentence-transformers/all-MiniLM-L6-v2") -> np.ndarray:
    model = SentenceTransformer(model_name)
    v = model.encode([q], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--pages", type=str, default=None, help='e.g. "3-4,6"')
    ap.add_argument("query", type=str)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    index, meta = _load_index(run_dir)

    # Optional page filter
    allowed_pages = _parse_pages(args.pages)
    ids = meta["ids"]
    metas = meta["metas"]

    if allowed_pages:
        mask = [i for i, m in enumerate(metas) if m.get("page") in allowed_pages]
        if not mask:
            print("No chunks match the page filter.")
            return

        # Build an IDMap to a subindex for filtering (simple approach)
        sub_vecs = []  # only load if we had stored vectors; we didn't
        # Simpler: we search global index then filter top hits post-hoc.
        # So we skip building subindex to keep code minimal.

    qv = _encode(args.query, model_name=meta.get("model") or "sentence-transformers/all-MiniLM-L6-v2")
    scores, I = index.search(qv, max(args.k, 10))  # overfetch, we will filter

    items: List[Tuple[float, int]] = []
    for rank in range(I.shape[1]):
        idx = int(I[0, rank])
        if idx < 0 or idx >= len(ids):
            continue
        if allowed_pages and metas[idx].get("page") not in allowed_pages:
            continue
        items.append((float(scores[0, rank]), idx))
        if len(items) >= args.k:
            break

    print(f"\nTop {len(items)} results for: '{args.query}'\n")
    for i, (score, idx) in enumerate(items, start=1):
        m = metas[idx]
        page = m.get("page")
        src = m.get("source_image")
        model = m.get("ocr_model")
        print(f"{i:02d}. score={score:.4f}  {src} | page {page} | model={model}")


if __name__ == "__main__":
    main()
