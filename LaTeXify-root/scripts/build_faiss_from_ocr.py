#!/usr/bin/env python3
"""
Build a FAISS index for an assessment from OCR JSONL.

Input JSONL schema (one record per text span):
{"page": 1, "text": "…", "bbox": [x0,y0,x1,y1]}  # extra keys are ignored

Output layout (sibling dir next to meta JSON):
<assessment.json>.index/
  ├─ faiss.index              (FAISS index, IP)
  ├─ embeddings.npy           (float32, N x D)
  └─ spans.jsonl              (JSONL; id-aligned with embeddings rows)

Model: sentence-transformers BGE-M3 (multi-lingual, retrieval-strong)
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np

def _read_jsonl(p: Path):
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True, help="e.g., dev/inputs/assessment.json")
    ap.add_argument("--ocr-jsonl", required=True, help="OCR JSONL path")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--model", default="BAAI/bge-m3")
    ap.add_argument("--normalize", type=int, default=1)
    args = ap.parse_args()

    meta_path = Path(args.meta).resolve()
    ocr_path  = Path(args.ocr_jsonl).resolve()
    out_dir   = meta_path.with_suffix(".json").with_suffix(".index")  # "<meta>.index"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Lazy import to avoid hard dep at import time
    try:
        from sentence_transformers import SentenceTransformer  #
    except Exception as e:
        raise RuntimeError("pip install sentence-transformers") from e

    try:
        import faiss  # type: ignore
    except Exception as e:
        raise RuntimeError("pip install faiss-cpu") from e

    spans = []
    texts = []
    for rec in _read_jsonl(ocr_path):
        t = (rec.get("text") or "").strip()
        if not t:
            continue
        spans.append(rec)
        texts.append(t)

    if not texts:
        raise RuntimeError(f"No OCR text found in {ocr_path}")

    model = SentenceTransformer(args.model)  # BGE-M3 model family is recommended for retrieval
    # BGE models expect instruction-free encode; see official tutorial for FAISS IO. :contentReference[oaicite:1]{index=1}
    embs = model.encode(texts, batch_size=args.batch, convert_to_numpy=True).astype("float32")
    if args.normalize:
        # cosine via inner product → L2 normalize for IP search
        embs /= (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)

    d = embs.shape[1]
    index = faiss.IndexFlatIP(d)  # exact IP for prototype scale; save/load with write/read_index. :contentReference[oaicite:2]{index=2}
    index.add(embs)

    np.save(out_dir / "embeddings.npy", embs)
    with (out_dir / "spans.jsonl").open("w", encoding="utf-8") as f:
        for rec in spans:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    faiss.write_index(index, str(out_dir / "faiss.index"))  # :contentReference[oaicite:3]{index=3}
    print(f"[index] wrote: {out_dir}/faiss.index  (N={len(spans)}, D={d})")

if __name__ == "__main__":
    main()
