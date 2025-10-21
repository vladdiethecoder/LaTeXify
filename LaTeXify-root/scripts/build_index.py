# scripts/build_index.py
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import List, Dict

import numpy as np
import faiss  # type: ignore
from sentence_transformers import SentenceTransformer

def _read_chunks(chunks_path: Path) -> List[Dict]:
    items: List[Dict] = []
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--batch_size", type=int, default=64)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    chunks_path = run_dir / "chunks.jsonl"
    if not chunks_path.exists():
        raise FileNotFoundError(f"No chunks.jsonl at {chunks_path}. Run scripts.build_chunks first.")

    chunks = _read_chunks(chunks_path)
    texts = [c.get("text", "") for c in chunks]
    model = SentenceTransformer(args.encoder)
    # This model maps sentences/paragraphs to 384-dim vectors (per its card).
    # We'll use normalized embeddings + IndexFlatIP.
    emb = model.encode(texts, batch_size=args.batch_size, normalize_embeddings=True, convert_to_numpy=True)
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    faiss_path = run_dir / "faiss.index"
    faiss.write_index(index, str(faiss_path))

    meta = {
        "encoder": args.encoder,
        "dim": dim,
        "count": len(chunks),
        "chunks_meta": [
            {
                "chunk_id": c["chunk_id"],
                "page_num": c["page_num"],
                "page_name": c["page_name"],
                "source_image": c["source_image"],
                "model": c.get("model", "unknown"),
                "start": c.get("start_char_in_page", 0),
                "end": c.get("end_char_in_page", 0),
            }
            for c in chunks
        ],
    }
    (run_dir / "faiss.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {faiss_path} and faiss.meta.json (dim={dim}, n={len(chunks)})")

if __name__ == "__main__":
    main()
