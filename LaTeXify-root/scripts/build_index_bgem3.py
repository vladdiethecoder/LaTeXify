#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_index_bgem3.py

Build a FAISS index from chunks produced by build_chunks.py using BGE-M3 embeddings.

- Input:  <run_dir>/<prefix>.chunks.jsonl  (list of {"id","text",...})
- Output: <run_dir>/<prefix>.faiss.index, <run_dir>/<prefix>.faiss.meta.json
          (or <out_dir>/faiss.* when --out is provided)

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
from typing import Any, Dict, List, Optional, Tuple

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


def _sanitize_prefix(raw: str) -> str:
    keep = [c if c.isalnum() or c in {"-", "_"} else "-" for c in raw.strip().lower()]
    prefix = "".join(keep).strip("-_")
    return prefix or "chunks"


def _prefix_from_chunks_file(path: Path) -> str:
    name = path.name
    if name.endswith(".chunks.jsonl"):
        return name[: -len(".chunks.jsonl")]
    return path.stem


def determine_chunks_path(run_path: Path, explicit_prefix: Optional[str]) -> Tuple[Path, str]:
    if explicit_prefix:
        prefix = _sanitize_prefix(explicit_prefix)
        chunk_path = run_path / f"{prefix}.chunks.jsonl"
        if not chunk_path.exists():
            raise SystemExit(f"[index] expected {chunk_path} (from --prefix {prefix})")
        return chunk_path, prefix

    legacy = run_path / "chunks.jsonl"
    if legacy.exists():
        if legacy.is_symlink():
            try:
                target = legacy.resolve(strict=True)
                return target, _prefix_from_chunks_file(target)
            except FileNotFoundError:
                pass
        return legacy, _prefix_from_chunks_file(legacy)

    candidates = sorted(p for p in run_path.glob("*.chunks.jsonl") if p.name != "chunks.jsonl")
    if len(candidates) == 1:
        chunk_path = candidates[0]
        return chunk_path, _prefix_from_chunks_file(chunk_path)

    if len(candidates) > 1:
        run_hint = _sanitize_prefix(run_path.name)
        preferred = [p for p in candidates if _prefix_from_chunks_file(p) == run_hint]
        if len(preferred) == 1:
            chunk_path = preferred[0]
            return chunk_path, _prefix_from_chunks_file(chunk_path)
        names = ", ".join(p.name for p in candidates)
        raise SystemExit(
            f"[index] multiple chunk files found in {run_path}: {names}. "
            "Pass --prefix to disambiguate."
        )

    raise SystemExit(f"[index] no *.chunks.jsonl found in {run_path}. Run build_chunks.py first.")


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

def resolve_output_paths(run_dir: str,
                         out: Optional[str],
                         prefix: Optional[str],
                         inferred_prefix: str) -> Tuple[Path, Path]:
    """
    Returns (index_path, meta_path).
    If prefix is provided, files are written inside run_dir using the prefix:
        <run_dir>/<prefix>.faiss.index, <run_dir>/<prefix>.faiss.meta.json
    Otherwise, fall back to ensure_out_dir semantics.
    """
    if prefix:
        run_path = Path(run_dir)
        run_path.mkdir(parents=True, exist_ok=True)
        index_path = run_path / f"{prefix}.faiss.index"
        meta_path = run_path / f"{prefix}.faiss.meta.json"
        return index_path, meta_path

    if not out and inferred_prefix and inferred_prefix != "chunks":
        run_path = Path(run_dir)
        run_path.mkdir(parents=True, exist_ok=True)
        index_path = run_path / f"{inferred_prefix}.faiss.index"
        meta_path = run_path / f"{inferred_prefix}.faiss.meta.json"
    else:
        out_dir = ensure_out_dir(run_dir, out)
        index_path = out_dir / "faiss.index"
        meta_path = out_dir / "faiss.meta.json"
    return index_path, meta_path

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


def _coerce_flags(flags: Any) -> Dict[str, bool]:
    if isinstance(flags, dict):
        return {str(k): bool(v) for k, v in flags.items()}
    if isinstance(flags, list):
        return {str(k): True for k in flags}
    if isinstance(flags, str):
        return {flags: True}
    return {}


def _pick_meta(chunk: Dict[str, Any], key: str) -> Any:
    if key in chunk and chunk[key] is not None:
        return chunk[key]
    meta = chunk.get("metadata")
    if isinstance(meta, dict):
        return meta.get(key)
    return None


def write_meta(meta_path: Path,
               model_name: str,
               dim: int,
               ids: List[str],
               chunks: List[Dict[str, Any]],
               run_dir: str) -> None:
    id_to_ref: List[Dict[str, Any]] = []
    for i, chunk in enumerate(chunks):
        meta = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else None
        id_to_ref.append({
            "id": ids[i],
            "text": chunk.get("text"),
            "page": _pick_meta(chunk, "page"),
            "bbox": _pick_meta(chunk, "bbox"),
            "block_type": _pick_meta(chunk, "block_type"),
            "label": _pick_meta(chunk, "label"),
            "labels": _pick_meta(chunk, "labels"),
            "page_span": _pick_meta(chunk, "page_span"),
            "pages": _pick_meta(chunk, "pages"),
            "block_ids": _pick_meta(chunk, "block_ids"),
            "semantic_id": _pick_meta(chunk, "semantic_id"),
            "flags": _coerce_flags(_pick_meta(chunk, "flags")),
            "source_backend": _pick_meta(chunk, "source_backend"),
            "source_backends": _pick_meta(chunk, "source_backends"),
            "metadata": meta,
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
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

def main():
    set_determinism(42)

    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="Per-PDF run dir containing chunks.jsonl")
    ap.add_argument("--out", default=None, help="Output directory for index (defaults inferred from run_dir)")
    ap.add_argument("--prefix", default=None,
                help="If set, write <run_dir>/<prefix>.faiss.index + .meta.json instead of directory outputs")
    ap.add_argument("--model_name", default="BAAI/bge-m3", help="Embedding model (default: BAAI/bge-m3)")
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    run_dir = args.run_dir
    run_path = Path(run_dir)
    explicit_prefix = args.prefix if args.prefix else None
    chunks_path, chunk_prefix = determine_chunks_path(run_path, explicit_prefix)
    index_path, meta_path = resolve_output_paths(run_dir, args.out, chunk_prefix if explicit_prefix else None, chunk_prefix)
    chunks = read_chunks(chunks_path)

    if not chunks:
        print("No chunks to index.")
        return

    texts = [chunks[i].get("text", "") for i in range(len(chunks))]
    ids = [ (chunks[i].get("id") if chunks[i].get("id") else f"chunk-{i:05d}") for i in range(len(chunks)) ]

    model = load_model(args.model_name)
    embeddings = embed_texts(model, texts, batch_size=args.batch_size)
    index = build_faiss(embeddings)

    faiss.write_index(index, str(index_path))
    write_meta(meta_path, args.model_name, embeddings.shape[1], ids, chunks, run_dir)

    print(f"[index] wrote {index_path}")
    print(f"[index] wrote {meta_path}")
    print(f"[index] dim={embeddings.shape[1]} count={len(ids)}")


if __name__ == "__main__":
    main()
