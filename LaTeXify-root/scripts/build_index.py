# scripts/build_index.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import faiss  # faiss-cpu
from sentence_transformers import SentenceTransformer
import numpy as np

def _read_chunks(chunks_path: Path) -> List[dict]:
    rows: List[dict] = []
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def _encode_texts(texts: List[str], model_name: str, device: str | None) -> np.ndarray:
    model = SentenceTransformer(model_name, device=device or "cpu")
    # produce L2-normalized embeddings (cosine-ready for IP index)
    embs = model.encode(
        texts,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,  # sentence-transformers handles L2 norm
        show_progress_bar=True,
    )
    return embs

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, type=str)
    ap.add_argument("--chunks", type=str, default=None, help="Optional path; defaults to <run_dir>/chunks.jsonl")
    ap.add_argument("--encoder", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                    help="Sentence-Transformers model id")
    ap.add_argument("--device", type=str, default=None, help="auto | cpu | cuda")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    chunks_path = Path(args.chunks) if args.chunks else (run_dir / "chunks.jsonl")
    if not chunks_path.exists():
        raise FileNotFoundError(f"No chunks.jsonl at {chunks_path}. Run scripts.build_chunks first.")

    rows = _read_chunks(chunks_path)
    if not rows:
        raise RuntimeError("chunks.jsonl is empty")

    texts = [r["text"] for r in rows]
    embs = _encode_texts(texts, model_name=args.encoder, device=args.device)

    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine via IP on normalized vectors
    index.add(embs)

    # persist
    faiss_path = run_dir / "faiss.index"
    faiss.write_index(index, str(faiss_path))

    meta = {
        "encoder": args.encoder,
        "dim": dim,
        "normalize": True,
        "count": len(rows),
        "chunks_path": str(chunks_path),
        "id_to_ref": [
            {
                "id": rows[i]["id"],
                "page": rows[i]["page"],
                "page_idx": rows[i].get("page_idx"),
                "char_start": rows[i]["char_start"],
                "char_end": rows[i]["char_end"],
            }
            for i in range(len(rows))
        ],
    }
    (run_dir / "faiss.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {faiss_path}")
    print(f"Wrote {run_dir/'faiss.meta.json'}")

if __name__ == "__main__":
    main()
