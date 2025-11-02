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

import json
import argparse
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

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


def _sanitize_prefix(raw: str) -> str:
    keep = [c if c.isalnum() or c in {"-", "_"} else "-" for c in raw.strip().lower()]
    prefix = "".join(keep).strip("-_")
    return prefix or "chunks"


def _prefix_from_chunks_file(path: Path) -> str:
    name = path.name
    if name.endswith(".chunks.jsonl"):
        return name[: -len(".chunks.jsonl")]
    return path.stem


def find_chunks_file(run_dir: Path) -> Path:
    legacy = run_dir / "chunks.jsonl"
    if legacy.exists():
        if legacy.is_symlink():
            try:
                return legacy.resolve(strict=True)
            except FileNotFoundError:
                pass
        return legacy

    candidates = sorted(p for p in run_dir.glob("*.chunks.jsonl") if p.name != "chunks.jsonl")
    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) > 1:
        run_hint = _sanitize_prefix(run_dir.name)
        preferred = [p for p in candidates if _prefix_from_chunks_file(p) == run_hint]
        if len(preferred) == 1:
            return preferred[0]
        names = ", ".join(p.name for p in candidates)
        raise SystemExit(
            f"Multiple chunk files found in {run_dir}: {names}. "
            "Move unwanted files or pass --prefix via build_chunks.py."
        )

    raise SystemExit(f"No *.chunks.jsonl files found in {run_dir}.")


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


@lru_cache(maxsize=None)
def _known_doc_classes(root: Path) -> Set[str]:
    hints: Set[str] = {
        "lix",
        "lix_article",
        "lix_textbook",
        "textbook",
        "novella",
        "newspaper",
        "contract",
        "article",
        "scrartcl",
    }
    classes_root = root / "kb" / "classes"
    if classes_root.exists():
        for cls_path in classes_root.rglob("*.cls"):
            stem = cls_path.stem.lower()
            hints.add(stem)
            hints.add(stem.replace("-", "_"))
            hints.add(f"lix_{stem}")
    return hints


def _infer_doc_class_from_json(blob: Any) -> Optional[str]:
    if not isinstance(blob, dict):
        return None
    for key in ("doc_class", "docClass", "document_class"):
        val = blob.get(key)
        if val:
            return str(val)
    plan = blob.get("plan")
    if isinstance(plan, dict):
        for key in ("doc_class", "docClass", "document_class"):
            val = plan.get(key)
            if val:
                return str(val)
    return None


def infer_doc_class(run_dir: Path) -> Optional[str]:
    """Best-effort doc_class inference from run metadata or directory hints."""
    candidate_jsons = [
        run_dir / "plan.json",
        run_dir / "metadata.json",
        run_dir / f"{run_dir.name}.json",
        run_dir / f"{run_dir.name}.meta.json",
        run_dir / f"{run_dir.name}.chunks_meta.json",
    ]
    seen: Set[Path] = set()
    for path in candidate_jsons + list(run_dir.glob("*.json")):
        if path in seen or not path.exists():
            continue
        seen.add(path)
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        doc_class = _infer_doc_class_from_json(blob)
        if doc_class:
            return str(doc_class)

    hints = _known_doc_classes(Path(__file__).resolve().parents[1])
    for part in reversed(run_dir.parts):
        lower = part.lower()
        if lower in hints or lower.startswith("lix_"):
            return lower
    return None


def _chunk_doc_class(chunk: Dict[str, Any], default: Optional[str]) -> Optional[str]:
    meta = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else None
    for source in (chunk, meta):
        if not isinstance(source, dict):
            continue
        for key in ("doc_class", "docClass", "document_class"):
            val = source.get(key)
            if val:
                return str(val)
    return default


def write_meta(run_dir: Path, ids: List[str], texts: List[str], chunks: List[dict], dim: int):
    default_doc_class = infer_doc_class(run_dir)
    id_to_ref: List[Dict[str, Any]] = []
    for cid, text, ch in zip(ids, texts, chunks):
        meta = ch.get("metadata") if isinstance(ch.get("metadata"), dict) else None
        doc_class = _chunk_doc_class(ch, default_doc_class)
        id_to_ref.append({
            "id": cid,
            "text": text,
            "page": _pick_meta(ch, "page"),
            "bbox": _pick_meta(ch, "bbox"),
            "block_type": _pick_meta(ch, "block_type"),
            "label": _pick_meta(ch, "label"),
            "labels": _pick_meta(ch, "labels"),
            "page_span": _pick_meta(ch, "page_span"),
            "pages": _pick_meta(ch, "pages"),
            "block_ids": _pick_meta(ch, "block_ids"),
            "semantic_id": _pick_meta(ch, "semantic_id"),
            "source_backend": _pick_meta(ch, "source_backend"),
            "source_backends": _pick_meta(ch, "source_backends"),
            "flags": _coerce_flags(_pick_meta(ch, "flags")),
            "doc_class": doc_class,
            "metadata": meta,
        })
    meta = {
        "dim": dim,
        "n": len(ids),
        "ids": ids,
        "id_to_ref": id_to_ref,
        "doc_class": default_doc_class,
        "run_dir": str(run_dir),
    }
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
    chunks_path = find_chunks_file(run_dir)
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
