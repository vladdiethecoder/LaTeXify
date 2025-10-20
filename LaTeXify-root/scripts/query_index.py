# scripts/query_index.py
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from FlagEmbedding import BGEM3FlagModel
import faiss

def _load_index(run_dir: Path):
    index = faiss.read_index(str(run_dir / "faiss.index"))
    meta = json.loads((run_dir / "faiss.meta.json").read_text(encoding="utf-8"))
    return index, meta

def _embed_query(q: str):
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    v = model.encode([q])["dense_vecs"][0].astype("float32")
    v = v / (np.linalg.norm(v) + 1e-12)
    return v

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("query", type=str)
    args = ap.parse_args()

    index, meta = _load_index(Path(args.run_dir))
    v = _embed_query(args.query)
    D, I = index.search(np.expand_dims(v, 0), args.k)

    rows = []
    for rank, (score, idx) in enumerate(zip(D[0].tolist(), I[0].tolist()), start=1):
        m = meta[idx]
        rows.append({"rank": rank, "score": float(score), **m})
    print(json.dumps(rows, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
