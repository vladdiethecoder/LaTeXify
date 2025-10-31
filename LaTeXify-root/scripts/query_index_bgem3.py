#!/usr/bin/env python
import json, argparse, numpy as np, faiss
from pathlib import Path
from FlagEmbedding import BGEM3FlagModel

p=argparse.ArgumentParser()
p.add_argument("--index", required=True)
p.add_argument("--meta", required=True)
p.add_argument("--queries", nargs="+", required=True)
p.add_argument("--model", default="BAAI/bge-m3")
p.add_argument("--k", type=int, default=5)
args=p.parse_args()

index = faiss.read_index(args.index)
meta = json.load(open(args.meta,"r",encoding="utf-8"))
idmap = {r["i"]: r["chunk_id"] for r in meta["rows"]}

enc = BGEM3FlagModel(args.model, use_fp16=True)
Q = enc.encode(args.queries, batch_size=8, max_length=8192)["dense_vecs"].astype("float32")
faiss.normalize_L2(Q)
D, I = index.search(Q, args.k)

for qi, q in enumerate(args.queries):
    print(f"\n=== Q{qi+1}: {q[:120]} ===")
    for rank,(score, idx) in enumerate(zip(D[qi], I[qi]),1):
        cid = idmap.get(int(idx), f"row{int(idx)}")
        print(f"{rank:2d}. {score:+.4f}  {cid}")
