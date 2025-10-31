#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_index_bgem3.py

Build a FAISS index from chunks produced by build_chunks.py using BGE-M3 embeddings.

- Input:  <run_dir>/chunks.jsonl  (list of {"id","text",...})
- Output: <out_dir>/faiss.index, <out_dir>/faiss.meta.json

Defaults:
- Model:  BAAI/bge-m3  (1024-dim)
- Index:  FAISS IndexFlatIP over L2-normalized vectors (cosine similarity)

Examples:
    python scripts/build_index_bgem3.py --run_dir dev/runs/user_e2e
    python scripts/build_index_bgem3.py --run_dir dev/runs/assessment_e2e --out indexes/assessment.index

Determinism:
- Sets seeds for random, numpy, and torch (if present).
"""
from __future__ import annotations

import os
import json
import argparse
import random
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import faiss  # type: ignore

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "sentence-transformers is required. Try: pip install sentence-transformers"
    ) from e

try:
    import torch
except Exception:
    torch = None  # type: ignore


def set_determinism(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        try:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True  # type: ignore
            torch.backends.cudnn.benchmark = False     # type: ignore
        except Exception:
            pass


def read_chunks(chunks_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not chunks_path.exists():
        return rows
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj.get("text"), str) and obj["text"].strip():
                    rows.append(obj)
            except Exception:
                continue
    return rows


def ensure_out_dir(run_dir: str, out: Optional[str]) -> Path:
    """
    Treat --out as a DIRECTORY always (directories may contain dots, e.g., *.index).
    If --out is None, infer from run_dir suffix.
    """
    if out:
        p = Path(out)
    else:
        rd = Path(run_dir).as_posix().lower()
        if rd.endswith("user_e2e"):
            p = Path("indexes/user.index")
        elif rd.endswith("assessment_e2e"):
            p = Path("indexes/assessment.index")
        elif rd.endswith("rubric_e2e"):
            p = Path("indexes/rubric.index")
        elif rd.endswith("assignment_e2e"):
            p = Path("indexes/assignment.index")
        else:
            p = Path(run_dir) / "index"
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_model(model_name: str = "BAAI/bge-m3") -> SentenceTransformer:
    device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
    model = SentenceTransformer(model_name, device=device)
    return model


def embed_texts(model: SentenceTransformer, texts: List[str], batch_size: int = 64) -> np.ndarray:
    prefixed = [f"passage: {t}" for t in texts]
    emb = model.encode(
        prefixed,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
    )
    assert isinstance(emb, np.ndarray)
    faiss.normalize_L2(emb)
    return emb.astype("float32", copy=False)


def build_faiss(embeddings: np.ndarray) -> faiss.Index:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def write_meta(out_dir: Path,
               model_name: str,
               dim: int,
               ids: List[str],
               chunks: List[Dict[str, Any]],
               run_dir: str) -> None:
    id_to_ref: List[Dict[str, Any]] = []
    for i, chunk in enumerate(chunks):
        id_to_ref.append({
            "id": ids[i],
            "text": chunk.get("text"),
            "page": chunk.get("page"),
            "bbox": chunk.get("bbox"),
            "block_type": chunk.get("block_type"),
            "semantic_id": chunk.get("semantic_id"),
            "flags": chunk.get("flags", {}),
        })

    meta = {
        "model": model_name,
        "dim": dim,
        "count": len(ids),
        "index_factory": "FlatIP",
        "normalize": True,
        "ids": ids,
        "id_to_ref": id_to_ref,
        "run_dir": run_dir,
    }
    (out_dir / "faiss.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main():
    set_determinism(42)

    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="Per-PDF run dir containing chunks.jsonl")
    ap.add_argument("--out", default=None, help="Output directory for index (defaults inferred from run_dir)")
    ap.add_argument("--model_name", default="BAAI/bge-m3", help="Embedding model (default: BAAI/bge-m3)")
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    run_dir = args.run_dir
    out_dir = ensure_out_dir(run_dir, args.out)
    chunks_path = Path(run_dir) / "chunks.jsonl"
    chunks = read_chunks(chunks_path)

    if not chunks:
        print("No chunks to index.")
        return

    texts = [chunks[i].get("text", "") for i in range(len(chunks))]
    ids = [ (chunks[i].get("id") if chunks[i].get("id") else f"chunk-{i:05d}") for i in range(len(chunks)) ]

    model = load_model(args.model_name)
    embeddings = embed_texts(model, texts, batch_size=args.batch_size)
    index = build_faiss(embeddings)

    faiss.write_index(index, str(out_dir / "faiss.index"))
    write_meta(out_dir, args.model_name, embeddings.shape[1], ids, chunks, run_dir)

    print(f"[index] wrote {out_dir / 'faiss.index'}")
    print(f"[index] wrote {out_dir / 'faiss.meta.json'}")
    print(f"[index] dim={embeddings.shape[1]} count={len(ids)}")


if __name__ == "__main__":
    main()
