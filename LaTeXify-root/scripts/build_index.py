# scripts/build_index.py
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

from FlagEmbedding import BGEM3FlagModel
import faiss

def _load_chunks(chunks_path: Path):
    rows = []
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def _embed_texts_bgem3(texts: list[str]):
    # BGE-M3 returns a dict with 'dense_vecs' (float32)
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)  # CPU OK; GPU if available
    embs = model.encode(texts, batch_size=16, max_length=8192)["dense_vecs"]
    return np.array(embs, dtype="float32")

def build_index(run_dir: Path):
    chunks_path = run_dir / "chunks.jsonl"
    if not chunks_path.exists():
        raise SystemExit(f"Missing chunks: {chunks_path}. Run the chunker first.")

    rows = _load_chunks(chunks_path)
    texts = [r["text"] for r in rows]
    meta = [{"chunk_id": r["chunk_id"], "page": r["page"], "label": r["label"], "bbox": r["bbox"]} for r in rows]

    vecs = _embed_texts_bgem3(texts)
    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product
    faiss.normalize_L2(vecs)        # normalize for cosine

    index.add(vecs)
    faiss.write_index(index, str(run_dir / "faiss.index"))
    (run_dir / "faiss.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True, help="dev/runs/<stamp>")
    args = ap.parse_args()
    build_index(Path(args.run_dir))

if __name__ == "__main__":
    main()
