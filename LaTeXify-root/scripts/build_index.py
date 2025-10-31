#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_index.py  (MiniLM 384-dim default)

Reads <run_dir>/chunks.jsonl and builds FAISS index:
  <run_dir>/faiss.index
  <run_dir>/faiss.meta.json

Also mirrors into:
  indexes/<role>/faiss.index
  indexes/<role>/faiss.meta.json

Where <role> is inferred from run_dir/pdf naming:
  user_e2e     -> user
  assessment_* -> assessment
  rubric_*     -> rubric
  else         -> assignment
"""
from __future__ import annotations

import os
import json
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
try:
    import faiss  # type: ignore
except Exception:
    import faiss_cpu as faiss  # type: ignore

from sentence_transformers import SentenceTransformer


ROLE_MAP = {
    "user": "user",
    "assessment": "assessment",
    "rubric": "rubric",
    "assignment": "assignment",
}

def infer_role(run_dir: str) -> str:
    s = run_dir.lower()
    for k, v in ROLE_MAP.items():
        if k in s:
            return v
    return "assignment"


def load_chunks(path: Path) -> List[dict]:
    out = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def embed_texts(texts: List[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> np.ndarray:
    model = SentenceTransformer(model_name)
    embs = model.encode(texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    return embs.astype("float32")


def build_faiss(embs: np.ndarray):
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)
    return index


def write_meta(run_dir: Path, ids: List[str], texts: List[str], chunks: List[dict], dim: int):
    id_to_ref = {}
    for cid, text, ch in zip(ids, texts, chunks):
        id_to_ref[cid] = {
            "id": cid,
            "text": text,
            "page": ch.get("page"),
            "bbox": ch.get("bbox"),
            "block_type": ch.get("block_type"),
            "semantic_id": ch.get("semantic_id"),
            "source_backend": ch.get("source_backend"),
            "flags": ch.get("flags", {}),
        }
    meta = {"dim": dim, "n": len(ids), "id_to_ref": id_to_ref}
    (run_dir / "faiss.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def mirror_to_indexes(run_dir: Path, role: str):
    role_dir = run_dir.parents[2] / "indexes" / role  # .../LaTeXify-root/indexes/<role>
    role_dir.mkdir(parents=True, exist_ok=True)
    # copy files
    (role_dir / "faiss.index").write_bytes((run_dir / "faiss.index").read_bytes())
    (role_dir / "faiss.meta.json").write_text((run_dir / "faiss.meta.json").read_text(encoding="utf-8"), encoding="utf-8")
    print(f"[{role}] -> {role_dir/'faiss.index'} ; {role_dir/'faiss.meta.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="Per-PDF run dir")
    ap.add_argument("--model_name", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    chunks_path = run_dir / "chunks.jsonl"
    chunks = load_chunks(chunks_path)
    if not chunks:
        print("No chunks to index.")
        raise SystemExit(1)

    texts = [c.get("text","") for c in chunks]
    ids = [c.get("id") or f"chunk-{i:05d}" for i, c in enumerate(chunks)]

    embs = embed_texts(texts, model_name=args.model_name)
    index = build_faiss(embs)
    faiss.write_index(index, str(run_dir / "faiss.index"))
    write_meta(run_dir, ids, texts, chunks, dim=embs.shape[1])

    role = infer_role(args.run_dir)
    mirror_to_indexes(run_dir, role)

    print(f"Wrote {run_dir/'faiss.index'} and {run_dir/'faiss.meta.json'}  (dim={embs.shape[1]}, n={len(ids)})")


if __name__ == "__main__":
    main()
