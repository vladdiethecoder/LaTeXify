# scripts/build_index_compat.py
from __future__ import annotations
import argparse, glob, json
from pathlib import Path
from typing import Dict, List, Tuple
import faiss, numpy as np
from sentence_transformers import SentenceTransformer

def _read_jsonl(path: Path) -> List[Dict]:
    out=[];
    if not path.exists(): return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                try: out.append(json.loads(line))
                except: pass
    return out

def _collect_chunks(run_dir: Path, chunks_glob: str|None, verbose: bool) -> List[Dict]:
    files=[]
    if chunks_glob:
        files += [Path(p) for p in glob.glob(str(run_dir / chunks_glob), recursive=True)]
    else:
        for pat in ["chunks/*.jsonl","chunks.jsonl"]:
            files += [Path(p) for p in glob.glob(str(run_dir / pat))]
    files = sorted({p.resolve() for p in files if p.exists()})
    if verbose: print(json.dumps({"candidate_files":[str(p) for p in files]}, indent=2))
    rows=[]
    for p in files:
        part=_read_jsonl(p)
        if verbose: print(json.dumps({"file":str(p),"lines":len(part)}, indent=2))
        rows += part
    return rows

def _rows_from_docs(run_dir: Path, verbose: bool) -> List[Dict]:
    docs = run_dir / "latex_docs.jsonl"
    rows = _read_jsonl(docs)
    if verbose: print(json.dumps({"fallback_docs_path":str(docs),"docs_lines":len(rows)}, indent=2))
    out=[]
    for r in rows:
        title=(r.get("title") or "").strip()
        question=(r.get("question") or "").strip()
        answer=(r.get("answer") or "").strip()
        code=r.get("code_blocks") or []
        code_excerpt="\n\n".join(code[:2]).strip()
        text="\n\n".join([t for t in [title,question,answer,code_excerpt] if t]).strip()
        if not text: continue
        cid=r.get("id") or f"kb_{abs(hash((title,question)))%10**9:09d}"
        out.append({"id":cid,"page":1,"label":"kb","text":text,
                    "source_image":r.get("url"),"ocr_model":r.get("source"),"bbox":[0,0,0,0]})
    return out

def _embed(texts: List[str], model: str) -> np.ndarray:
    enc = SentenceTransformer(model)
    return enc.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype("float32")

def build_index(run_dir: Path, model: str, chunks_glob: str|None, verbose: bool) -> Tuple[Path,Path]:
    rows = _collect_chunks(run_dir, chunks_glob, verbose)
    if not rows:
        if verbose: print(json.dumps({"info":"no_chunks_found_try_docs_fallback"}, indent=2))
        rows = _rows_from_docs(run_dir, verbose)
    clean=[]
    for r in rows:
        cid=str(r.get("id") or "").strip()
        txt=(r.get("text") or "").strip()
        if cid and txt: clean.append((cid, txt, r))
    if verbose:
        print(json.dumps({"rows_total":len(rows),"rows_clean":len(clean)}, indent=2))
        if rows: print(json.dumps({"first_row_keys":sorted(rows[0].keys())}, indent=2))
    if not clean: raise SystemExit("No chunks to index (compat).")

    clean.sort(key=lambda t: t[0])
    ids=[c[0] for c in clean]; texts=[c[1] for c in clean]
    metas=[{"id":c[2].get("id"),"page":c[2].get("page"),"label":c[2].get("label"),
            "source_image":c[2].get("source_image"),"ocr_model":c[2].get("ocr_model"),
            "bbox":c[2].get("bbox")} for c in clean]

    X=_embed(texts, model); d=X.shape[1]
    index=faiss.IndexFlatIP(d); index.add(X)
    idx_p=run_dir/"faiss.index"; idx_p.write_bytes(faiss.serialize_index(index))
    meta_p=run_dir/"faiss.meta.json"
    meta_p.write_text(json.dumps({"dim":d,"size":len(texts),"ids":ids,"metas":metas,"model":model}, indent=2), encoding="utf-8")
    (run_dir/"latex_docs.index").write_bytes(idx_p.read_bytes())
    (run_dir/"latex_docs.meta.json").write_text(meta_p.read_text(encoding="utf-8"), encoding="utf-8")
    return idx_p, meta_p

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=Path, required=True)
    ap.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--chunks_glob", type=str, default=None)
    ap.add_argument("--verbose", action="store_true")
    a=ap.parse_args()
    idx, meta = build_index(a.run_dir, a.model, a.chunks_glob, a.verbose)
    print(json.dumps({"index":str(idx),"meta":str(meta)}, indent=2))

if __name__=="__main__":
    main()
