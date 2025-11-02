#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
retrieval_agent.py (v1.1)

Build Context Bundles per task using:
- semantic_id anchor matches, and
- FAISS dense retrieval + reranking (BGE-M3 + cross-encoder, MMR fallback).

Determinism: fixed seeds; cosine via L2-normalized vectors + IndexFlatIP.

Usage:
  # all tasks
  python scripts/retrieval_agent.py \
    --plan build/plan.json --all_tasks \
    --assessment indexes/assessment.index \
    --rubric indexes/rubric.index \
    --assignment indexes/assignment.index \
    --user indexes/user.index \
    --out_dir build/context

  # options
  --k 6 --mmr_lambda 0.6 --mmr_pool 20 --reranker BAAI/bge-reranker-v2-m3
"""
from __future__ import annotations

import os
import re
import json
import faiss  # type: ignore
import argparse
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np

from scripts.rerankers import get_reranker, BaseReranker

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:  # pragma: no cover
    raise RuntimeError("pip install sentence-transformers") from e

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


def load_plan(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _meta_path(p: str) -> Path:
    P = Path(p)
    if P.is_dir():
        return P / "faiss.meta.json"
    if P.name == "faiss.index":
        return P.with_name("faiss.meta.json")
    return P


def load_meta(idx: str) -> dict:
    mpath = _meta_path(idx)
    meta = json.loads(mpath.read_text(encoding="utf-8"))
    print(f"[retrieval] using meta: {mpath}")
    return meta


def load_index(idx_dir: str) -> faiss.Index:
    p = Path(idx_dir)
    if p.is_file() and p.name == "faiss.index":
        return faiss.read_index(str(p))
    if p.is_dir():
        return faiss.read_index(str(p / "faiss.index"))
    # if they passed meta path, go sibling
    if p.name == "faiss.meta.json":
        return faiss.read_index(str(p.with_name("faiss.index")))
    raise FileNotFoundError(f"Cannot locate faiss.index for {idx_dir}")


def load_model(name: str = "BAAI/bge-m3") -> SentenceTransformer:
    device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
    return SentenceTransformer(name, device=device)


def _encode_query(model: SentenceTransformer, text: str) -> np.ndarray:
    # BGE-M3 no longer requires query instruction; embed raw query.
    q = model.encode([text], convert_to_numpy=True, show_progress_bar=False)
    faiss.normalize_L2(q)
    return q.astype("float32", copy=False)


def _encode_passages(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    # Index was built on normalized "passage" embeddings; encode candidates similarly.
    X = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    faiss.normalize_L2(X)
    return X.astype("float32", copy=False)


def _search(index: faiss.Index, query: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
    # returns (scores, ids) with cosine/IP scores (vectors normalized)
    D, I = index.search(query, topk)
    return D[0], I[0]


def _gather_candidates(meta: dict, ids: np.ndarray) -> List[dict]:
    # id_to_ref can be a list (positional) or dict keyed by chunk id.
    id_to_ref = meta.get("id_to_ref", [])
    out: List[dict] = []
    if isinstance(id_to_ref, list):
        for pos in ids:
            if 0 <= int(pos) < len(id_to_ref):
                out.append(id_to_ref[int(pos)])
        return out

    if isinstance(id_to_ref, dict):
        ordered_ids = meta.get("ids")
        if isinstance(ordered_ids, list):
            for pos in ids:
                if 0 <= int(pos) < len(ordered_ids):
                    ref = id_to_ref.get(ordered_ids[int(pos)])
                    if ref:
                        out.append(ref)
        else:
            for pos in ids:
                ref = id_to_ref.get(str(int(pos))) or id_to_ref.get(int(pos))
                if ref:
                    out.append(ref)
    return out


def mmr_select(doc_vecs: np.ndarray,
               q_vec: np.ndarray,
               k: int = 6,
               lam: float = 0.6) -> List[int]:
    """
    Simple MMR: greedily pick up to k items maximizing
      lam * sim(q, d) - (1 - lam) * max_j sim(d, selected_j)
    with cosine/IP similarities (vectors normalized).
    """
    sims_q = (doc_vecs @ q_vec.reshape(-1))
    selected: List[int] = []
    candidates = list(range(doc_vecs.shape[0]))
    while candidates and len(selected) < k:
        best_i = None
        best_score = -1e9
        for i in candidates:
            if not selected:
                redundancy = 0.0
            else:
                redundancy = max((doc_vecs[i] @ doc_vecs[j]) for j in selected)
            score = lam * sims_q[i] - (1.0 - lam) * redundancy
            if score > best_score:
                best_score = score
                best_i = i
        selected.append(best_i)  # type: ignore
        candidates.remove(best_i)  # type: ignore
    return selected


def filter_by_semantic(meta: dict, sid: str) -> List[dict]:
    sidU = sid.upper()
    out = []
    for row in meta.get("id_to_ref", []):
        rsid = str(row.get("semantic_id", "")).upper()
        if rsid == sidU or (len(sidU) >= 2 and (rsid.startswith(sidU) or sidU.startswith(rsid))):
            out.append(row)
    return out


def _assignment_headers(meta: dict, k: int = 4) -> List[dict]:
    def is_hdr(s: str) -> bool:
        return bool(re.search(r"\b(instruction|submission|format|policy|overview|guideline|academic|grading)\b",
                              s or "", re.I))
    rows = [r for r in meta.get("id_to_ref", []) if is_hdr(r.get("semantic_id") or r.get("text") or "")]
    if not rows:
        rows = meta.get("id_to_ref", [])[:k]
    return rows[:k]


def _text(r: dict) -> str:
    return str(r.get("text") or "")


def build_for_task(task_id: str,
                   model: SentenceTransformer,
                   assess_idx: faiss.Index, assess_meta: dict,
                   rubric_idx: faiss.Index, rubric_meta: dict,
                   assignment_idx: faiss.Index, assignment_meta: dict,
                   user_idx: faiss.Index, user_meta: dict,
                   reranker: BaseReranker,
                   k: int, mmr_pool: int, mmr_lambda: float,
                   plan_doc_class: Optional[str]) -> dict:
    # Seed anchor rows via semantic_id
    q_rows = filter_by_semantic(assess_meta, task_id)
    question_text = " ".join(_text(r) for r in q_rows)[:4000] if q_rows else task_id

    # Dense query from anchor/question
    q_vec = _encode_query(model, question_text)

    # Search each index, gather a pool, rerank (fallback to MMR)
    plan_doc_class_norm = (plan_doc_class or "").strip().lower()

    def _matches_doc_class(row: dict) -> bool:
        if not plan_doc_class_norm:
            return True
        cand = row.get("doc_class")
        if not cand and isinstance(row.get("metadata"), dict):
            cand = row["metadata"].get("doc_class")
        if not cand:
            return True
        return str(cand).strip().lower() == plan_doc_class_norm

    def pool_pick(idx: faiss.Index, meta: dict) -> List[dict]:
        pool_size = max(mmr_pool, k, 20)
        _, ids = _search(idx, q_vec, topk=pool_size)
        cands = [r for r in _gather_candidates(meta, ids) if _matches_doc_class(r)]
        if not cands:
            return []
        reranked = reranker.rerank(question_text, cands, top_k=k)
        if reranker.available():
            return [s.payload for s in reranked]
        doc_vecs = _encode_passages(model, [_text(r) for r in cands])
        sel = mmr_select(doc_vecs, q_vec[0], k=min(k, len(cands)), lam=mmr_lambda)
        return [cands[i] for i in sel]

    assess_sel = pool_pick(assess_idx, assess_meta)
    rubric_sel = pool_pick(rubric_idx, rubric_meta)
    assign_sel = _assignment_headers(assignment_meta) or pool_pick(assignment_idx, assignment_meta)
    user_sel = pool_pick(user_idx, user_meta)

    # Flags from user
    flags = []
    for r in user_sel:
        f = r.get("flags") or {}
        if f.get("low_confidence") or f.get("high_ocr_disagreement"):
            flags.append({"page": r.get("page"), "span": "", "reason": "uncertain OCR"})

    bundle = {
        "task_id": task_id,
        "question": {
            "text": " ".join(_text(r) for r in (assess_sel or q_rows))[:4000],
            "meta": {
                "anchors": [task_id],
                "pages": list({r.get("page") for r in (assess_sel or q_rows) if r.get("page") is not None})
            }
        },
        "rubric": {
            "criteria": [
                {"id": f"R{i+1}", "label": (r.get("semantic_id") or "criterion"),
                 "verbatim": _text(r)[:2000]}
                for i, r in enumerate(rubric_sel if rubric_sel else rubric_meta.get("id_to_ref", [])[:k])
            ]
        },
        "assignment_rules": [_text(r)[:1000] for r in assign_sel[:k]],
        "user_answer": {
            "chunks": [{"page": r.get("page"), "text": _text(r)[:1200]} for r in user_sel],
            "flags": flags
        }
    }
    return bundle


def collect_task_ids(plan: dict) -> List[str]:
    return [t["id"] for t in plan.get("tasks", []) if t.get("type") == "question"]


def main():
    set_determinism(42)
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--task_id", default=None)
    ap.add_argument("--all_tasks", action="store_true")
    ap.add_argument("--assessment", required=True)
    ap.add_argument("--rubric", required=True)
    ap.add_argument("--assignment", required=True)
    ap.add_argument("--user", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--k", type=int, default=6, help="MMR top-k per source")
    ap.add_argument("--mmr_pool", type=int, default=20, help="FAISS pool size before rerank/MMR fallback")
    ap.add_argument("--mmr_lambda", type=float, default=0.6, help="MMR relevance weight")
    ap.add_argument("--reranker", default="BAAI/bge-reranker-v2-m3", help="Cross-encoder reranker model name or 'none'")
    args = ap.parse_args()

    plan = load_plan(args.plan)

    assess_meta = load_meta(args.assessment)
    rubric_meta = load_meta(args.rubric)
    assignment_meta = load_meta(args.assignment)
    user_meta = load_meta(args.user)

    assess_idx = load_index(args.assessment)
    rubric_idx = load_index(args.rubric)
    assignment_idx = load_index(args.assignment)
    user_idx = load_index(args.user)

    model = load_model("BAAI/bge-m3")
    reranker = get_reranker(args.reranker)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    task_ids = [args.task_id] if args.task_id and not args.all_tasks else collect_task_ids(plan)
    for tid in task_ids:
        bundle = build_for_task(
            tid, model,
            assess_idx, assess_meta,
            rubric_idx, rubric_meta,
            assignment_idx, assignment_meta,
            user_idx, user_meta,
            reranker=reranker,
            k=args.k, mmr_pool=args.mmr_pool, mmr_lambda=args.mmr_lambda,
            plan_doc_class=plan.get("doc_class")
        )
        (out_dir / f"{tid}.json").write_text(json.dumps(bundle, indent=2), encoding="utf-8")
        print(f"[retrieval] wrote {out_dir / (tid + '.json')}")

if __name__ == "__main__":
    main()
