# scripts/build_index.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def _load_chunks(chunks_path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            # normalize keys
            if "chunk_id" in rec and "id" not in rec:
                rec["id"] = rec["chunk_id"]
            if "text" not in rec:
                rec["text"] = ""
            rows.append(rec)
    return rows


def _build_embeddings(texts: List[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> np.ndarray:
    model = SentenceTransformer(model_name)
    embs = model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    return embs.astype("float32")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="Path to dev/runs/<STAMP>")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    chunks_path = run_dir / "chunks.jsonl"
    if not chunks_path.exists():
        raise FileNotFoundError(f"No chunks.jsonl at {chunks_path}. Run scripts.build_chunks first.")

    rows = _load_chunks(chunks_path)
    if not rows:
        raise RuntimeError("No chunks to index.")

    ids: List[str] = []
    metas: List[Dict] = []
    texts: List[str] = []
    for r in rows:
        cid = r.get("id") or r.get("chunk_id")
        txt = (r.get("text") or "").strip()
        if not cid or not txt:
            # skip truly empty entries
            continue
        ids.append(str(cid))
        metas.append({
            "id": str(cid),
            "page": r.get("page"),
            "label": r.get("label"),
            "source_image": r.get("source_image"),
            "ocr_model": r.get("ocr_model"),
            "bbox": r.get("bbox"),
        })
        texts.append(txt)

    if not texts:
        raise RuntimeError("All chunks are empty; nothing to index.")

    embs = _build_embeddings(texts)
    dim = embs.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(embs)

    faiss.write_index(index, str(run_dir / "faiss.index"))
    meta = {
        "dim": dim,
        "size": len(texts),
        "ids": ids,
        "metas": metas,
        "model": "sentence-transformers/all-MiniLM-L6-v2",
    }
    (run_dir / "faiss.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {run_dir/'faiss.index'} and {run_dir/'faiss.meta.json'} (dim={dim}, n={len(texts)})")


if __name__ == "__main__":
    main()
