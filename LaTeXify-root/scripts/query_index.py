# scripts/query_index.py
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None  # offline tests can run without network


# ---------------------------
# Core index I/O
# ---------------------------
def _load_index(run_dir: Path) -> Tuple[faiss.Index, Dict]:
    idx_p = run_dir / "faiss.index"
    meta_p = run_dir / "faiss.meta.json"
    if not idx_p.exists() or not meta_p.exists():
        raise SystemExit(f"Missing index/meta in: {run_dir}")
    index = faiss.read_index(str(idx_p))
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    return index, meta


# ---------------------------
# Deterministic encoders + dim handling
# ---------------------------
def _hash_vec(text: str, dim: int) -> np.ndarray:
    """Deterministic, no-network fallback (for tests/CI)."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
    v = rng.normal(size=(1, dim)).astype("float32")
    v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    return v


def _encode(q: str, model_name: str, dim: int) -> np.ndarray:
    """
    Encode a single query into a normalized vector of shape (1, dim).
    - If DISABLE_ST_EMBEDDING=1 or SentenceTransformer is unavailable or model_name=='dummy',
      we use a deterministic hash vector of the requested dim.
    - Otherwise we use SentenceTransformer, then coerce to match dim (truncate or zero-pad).
    """
    disable = os.getenv("DISABLE_ST_EMBEDDING") == "1"
    if disable or model_name.lower() == "dummy" or SentenceTransformer is None:
        return _hash_vec(q, dim)

    model = SentenceTransformer(model_name)
    v = model.encode([q], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    # coerce dimension if needed
    if v.shape[1] == dim:
        return v
    if v.shape[1] > dim:
        return v[:, :dim]
    # pad zeros deterministically
    pad = np.zeros((1, dim - v.shape[1]), dtype="float32")
    return np.concatenate([v, pad], axis=1)


# ---------------------------
# Helpers: stable ranking + MMR
# ---------------------------
def _reconstruct_many(index: faiss.Index, ids: List[int]) -> np.ndarray:
    vecs = []
    for i in ids:
        try:
            vec = faiss.vector_float_to_array(index.reconstruct(i))
        except Exception:
            # Not all indices support reconstruct; degrade gracefully
            return np.zeros((0, getattr(index, "d", 0)), dtype="float32")
        vecs.append(vec)
    return np.vstack(vecs).astype("float32") if vecs else np.zeros((0, getattr(index, "d", 0)), dtype="float32")


def _stable_pairs(scores: np.ndarray, I: np.ndarray) -> List[Tuple[float, int]]:
    items: List[Tuple[float, int]] = []
    for rank in range(I.shape[1]):
        idx = int(I[0, rank])
        sc = float(scores[0, rank])
        if idx >= 0:
            items.append((sc, idx))
    # stable sort by (-score, idx)
    items.sort(key=lambda t: (-t[0], t[1]))
    return items


def _mmr_select(index: faiss.Index, cand_ids: List[int], qv: np.ndarray, k: int, lambda_coef: float = 0.5) -> List[int]:
    """
    Deterministic Maximal Marginal Relevance selection on reconstructed vectors.
    If reconstruct() unsupported, returns first k candidates (already stably sorted).
    """
    if not cand_ids:
        return []
    X = _reconstruct_many(index, cand_ids)  # [n, d]
    if X.size == 0:
        return cand_ids[:k]
    q = qv.reshape(-1).astype("float32")
    q_sims = X @ q  # cosine/IP (assumes normalized)
    selected: List[int] = []
    remaining = list(range(len(cand_ids)))
    while remaining and len(selected) < k:
        if not selected:
            j = int(np.argmax(q_sims[remaining]))
            selected.append(remaining.pop(j))
            continue
        sel_vecs = X[selected]
        rem_vecs = X[remaining]
        red = rem_vecs @ sel_vecs.T
        max_red = red.max(axis=1) if red.size else np.zeros((len(remaining),), dtype="float32")
        mmr = lambda_coef * q_sims[remaining] - (1.0 - lambda_coef) * max_red
        j = int(np.argmax(mmr))
        selected.append(remaining.pop(j))
    return [cand_ids[i] for i in selected]


# ---------------------------
# Public: build_context_bundle
# ---------------------------
def build_context_bundle(task: Dict, indices: Dict[str, str], k_user: int = 6) -> Dict:
    """
    task: {"id": "...", "question": "..."}
    indices: {"assignment": "<dir>", "assessment": "<dir>", "rubric": "<dir>", "user": "<dir>"}
    Returns bundle and writes evidence/{task_id}.json with retrieval traces.
    """
    task_id = (task.get("id") or "T00").strip()
    question = (task.get("question") or "").strip()

    bundle = {
        "question": question,
        "rubric": [],
        "assignment_rules": [],
        "assessment": [],
        "user_answer": {"chunks": [], "flags": []},
    }
    traces = {"event": "context_bundle", "task_id": task_id, "question": question, "hits": {}}

    def _search(dir_path: str, k: int) -> List[Dict]:
        run_dir = Path(dir_path)
        index, meta = _load_index(run_dir)
        model_name = (meta.get("model") or "BAAI/bge-m3").strip()
        qv = _encode(question, model_name=model_name, dim=index.d)
        # overfetch for MMR
        scores, I = index.search(qv, max(k * 5, 20))
        pairs = _stable_pairs(scores, I)
        cands = [j for _, j in pairs]
        chosen = _mmr_select(index, cands, qv[0], k)
        out: List[Dict] = []
        # Build rank list; use scores consistent with I/pairs
        score_map = {idx: float(sc) for sc, idx in pairs}
        for rank, j in enumerate(chosen, start=1):
            m = meta["metas"][j] if 0 <= j < len(meta.get("metas", [])) else {}
            out.append({"rank": rank, "score": score_map.get(j, 0.0), "meta": m})
        return out

    for key, k in [("rubric", 8), ("assignment", 8), ("assessment", 6)]:
        dirp = indices.get(key)
        if dirp:
            hits = _search(dirp, k)
            if key == "rubric":
                bundle["rubric"] = hits
            elif key == "assignment":
                bundle["assignment_rules"] = hits
            else:
                bundle["assessment"] = hits
            traces["hits"][key] = hits

    if indices.get("user"):
        uhits = _search(indices["user"], k_user)
        bundle["user_answer"]["chunks"] = uhits
        if len(uhits) < k_user:
            bundle["user_answer"]["flags"].append("insufficient_user_context")
        traces["hits"]["user"] = uhits

    # write evidence
    Path("evidence").mkdir(exist_ok=True)
    (Path("evidence") / f"{task_id}.json").write_text(
        json.dumps(traces, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return bundle


# ---------------------------
# Legacy CLI (ad-hoc query) + Bundle mode
# ---------------------------
def _print_top(index: faiss.Index, meta: Dict, query: str, k: int, pages: Optional[str]):
    model_name = meta.get("model") or "BAAI/bge-m3"
    qv = _encode(query, model_name=model_name, dim=index.d)
    scores, I = index.search(qv, k)
    metas = meta.get("metas", [])
    print(f"\nTop {k} results for: '{query}'\n")
    for i in range(k):
        idx = int(I[0, i])
        if idx < 0:
            continue
        m = metas[idx] if 0 <= idx < len(metas) else {}
        print(f"{i+1:02d}. score={scores[0, i]:.4f} | page={m.get('page')} | src={m.get('source_image')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=False, help="Index dir for ad-hoc queries (legacy).")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--pages", type=str, default=None, help='e.g. "3-4,6"')
    ap.add_argument("--bundle", action="store_true", help="Read payload on stdin and emit a Context Bundle.")
    ap.add_argument("query", type=str, nargs="?")
    args = ap.parse_args()

    if args.bundle:
        payload = json.loads(sys.stdin.read())
        bundle = build_context_bundle(payload["task"], payload["indices"], k_user=int(payload.get("k_user", 6)))
        print(json.dumps(bundle, ensure_ascii=False, indent=2))
        return

    if not args.run_dir or not args.query:
        raise SystemExit("Usage: -m scripts.query_index --run_dir <dir> 'your query'  OR  --bundle with stdin payload")

    run_dir = Path(args.run_dir)
    index, meta = _load_index(run_dir)
    _print_top(index, meta, args.query, args.k, args.pages)


if __name__ == "__main__":
    main()
