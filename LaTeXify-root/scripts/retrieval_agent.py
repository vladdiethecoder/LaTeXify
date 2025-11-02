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
  --config scripts/retrieval_config.json \
  --k 6 --stage1_topk 24 --mmr_lambda 0.6 \
  --reranker BAAI/bge-reranker-v2-m3
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


DEFAULT_CONFIG: Dict[str, Any] = {
    "vector": {"top_k_stage1": 24},
    "reranker": {
        "model": "BAAI/bge-reranker-v2-m3",
        "top_k_stage2": 6,
        "device": None,
        "batch_size": 8,
    },
    "fallback": {"strategy": "mmr", "lambda": 0.6},
}


def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_dicts(result[key], value)  # type: ignore[arg-type]
        else:
            result[key] = value
    return result


def load_config(path: Optional[str]) -> Dict[str, Any]:
    cfg_path: Optional[Path] = None
    if path:
        cfg_path = Path(path)
    else:
        default_path = Path(__file__).with_name("retrieval_config.json")
        if default_path.exists():
            cfg_path = default_path

    if cfg_path and cfg_path.exists():
        try:
            text = cfg_path.read_text(encoding="utf-8")
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                try:
                    import yaml  # type: ignore
                except Exception as exc:  # pragma: no cover - optional dep
                    raise RuntimeError(
                        "PyYAML is required to load YAML retrieval configs"
                    ) from exc
                data = yaml.safe_load(text) or {}
        except Exception as e:
            print(f"[retrieval] warning: could not load config {cfg_path}: {e}")
            data = {}
        if isinstance(data, dict):
            return _merge_dicts(DEFAULT_CONFIG, data)
        return dict(DEFAULT_CONFIG)
    return dict(DEFAULT_CONFIG)


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


def _compact_entry(row: dict,
                   *,
                   max_len: int = 1000,
                   include_label: bool = False) -> Optional[dict]:
    text = _text(row).strip()
    if not text:
        return None
    entry: Dict[str, Any] = {"text": text[:max_len]}
    if row.get("page") is not None:
        entry["page"] = row.get("page")
    if include_label:
        label = row.get("semantic_id") or row.get("label")
        if label:
            entry["label"] = label
    doc_class = row.get("doc_class")
    if not doc_class and isinstance(row.get("metadata"), dict):
        doc_class = row["metadata"].get("doc_class")
    if doc_class:
        entry["doc_class"] = doc_class
    return entry


def _compact_list(rows: List[dict], *, max_len: int, include_label: bool = False) -> List[dict]:
    out: List[dict] = []
    for row in rows:
        entry = _compact_entry(row, max_len=max_len, include_label=include_label)
        if entry:
            out.append(entry)
    return out


def build_for_task(task_id: str,
                   model: SentenceTransformer,
                   assess_idx: faiss.Index, assess_meta: dict,
                   rubric_idx: faiss.Index, rubric_meta: dict,
                   assignment_idx: faiss.Index, assignment_meta: dict,
                   user_idx: faiss.Index, user_meta: dict,
                   reranker: BaseReranker,
                   stage1_topk: int,
                   stage2_topk: int,
                   fallback_strategy: str,
                   fallback_lambda: float,
                   plan_doc_class: Optional[str]) -> dict:
    # Seed anchor rows via semantic_id
    q_rows = filter_by_semantic(assess_meta, task_id)
    question_text = " ".join(_text(r) for r in q_rows)[:4000] if q_rows else task_id

    # Dense query from anchor/question
    q_vec = _encode_query(model, question_text)

    # Search each index, gather a pool, rerank (fallback to MMR)
    plan_doc_class_norm = (plan_doc_class or "").strip().lower()
    fallback_kind = (fallback_strategy or "mmr").lower()
    final_k = stage2_topk if stage2_topk > 0 else stage1_topk
    final_k = max(final_k, 0)
    pool_fetch = max(stage1_topk, final_k, 1)
    use_reranker = reranker.available() and stage2_topk > 0

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
        _, ids = _search(idx, q_vec, topk=pool_fetch)
        cands = [r for r in _gather_candidates(meta, ids) if _matches_doc_class(r)]
        if not cands:
            return []
        limit = min(final_k if final_k > 0 else len(cands), len(cands))
        if limit <= 0:
            return []
        if use_reranker:
            reranked = reranker.rerank(question_text, cands, top_k=limit)
            payloads = [s.payload for s in reranked if s.payload]
            if payloads:
                return payloads[:limit]
        if fallback_kind == "mmr":
            doc_vecs = _encode_passages(model, [_text(r) for r in cands])
            if getattr(doc_vecs, "size", 0) > 0:
                sel = mmr_select(doc_vecs, q_vec[0], k=min(limit, len(cands)), lam=fallback_lambda)
                return [cands[i] for i in sel]
        return cands[:limit]

    assess_sel = pool_pick(assess_idx, assess_meta)
    rubric_sel = pool_pick(rubric_idx, rubric_meta)
    assign_limit = max(final_k, 1)
    assign_sel = _assignment_headers(assignment_meta, k=assign_limit) or pool_pick(assignment_idx, assignment_meta)
    user_sel = pool_pick(user_idx, user_meta)

    # Flags from user
    flag_notes: List[Dict[str, Any]] = []
    for r in user_sel:
        f = r.get("flags") or {}
        if f.get("low_confidence") or f.get("high_ocr_disagreement"):
            flag_notes.append({"page": r.get("page"), "reason": "uncertain OCR"})

    flags: Dict[str, Any] = {"ocr_uncertain": bool(flag_notes)}
    if flag_notes:
        flags["notes"] = flag_notes

    rubric_source = rubric_sel if rubric_sel else rubric_meta.get("id_to_ref", [])[:max(final_k, 0)]
    assignment_entries = _compact_list(assign_sel[:assign_limit], max_len=800)
    user_entries = _compact_list(user_sel, max_len=1200)
    assessment_entries = _compact_list(assess_sel or q_rows, max_len=1200)
    rubric_entries = _compact_list(rubric_source, max_len=1200, include_label=True)

    bundle = {
        "task_id": task_id,
        "doc_class": plan_doc_class,
        "question": " ".join(_text(r) for r in (assess_sel or q_rows))[:4000] or task_id,
        "assessment": assessment_entries,
        "rubric": rubric_entries,
        "assignment_rules": assignment_entries,
        "user_answer": {
            "chunks": user_entries,
            "flags": flags
        },
        "diagnostics": {
            "vector_top_k": stage1_topk,
            "stage2_top_k": stage2_topk,
            "fallback_strategy": fallback_kind,
            "reranker": {
                "name": getattr(reranker, "name", "none"),
                "available": reranker.available(),
                "status": reranker.status() if hasattr(reranker, "status") else "unknown",
                "used": use_reranker,
            },
        },
    }
    return bundle


def collect_task_ids(plan: dict) -> List[str]:
    task_ids: List[str] = []
    for task in plan.get("tasks", []):
        if not isinstance(task, dict) or "id" not in task:
            continue
        kind = (task.get("kind") or "").lower()
        if kind in {"preamble", "titlepage"}:
            continue
        if kind:
            task_ids.append(task["id"])
            continue
        if task.get("type") == "question":
            task_ids.append(task["id"])
    return task_ids


def main():
    set_determinism(42)
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--config", default=None, help="Path to retrieval config (JSON or YAML).")
    ap.add_argument("--task_id", default=None)
    ap.add_argument("--all_tasks", action="store_true")
    ap.add_argument("--assessment", required=True)
    ap.add_argument("--rubric", required=True)
    ap.add_argument("--assignment", required=True)
    ap.add_argument("--user", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--stage1_topk", type=int, default=None, help="Override vector search top_k before rerank.")
    ap.add_argument("--k", type=int, default=None, help="Override reranker top_k after rerank.")
    ap.add_argument("--mmr_pool", type=int, default=None, help="Deprecated alias for --stage1_topk.")
    ap.add_argument("--mmr_lambda", type=float, default=None, help="Override fallback lambda for MMR.")
    ap.add_argument("--reranker", default=None, help="Cross-encoder reranker model name or 'none'.")
    ap.add_argument("--reranker_device", default=None, help="Override reranker device (cpu/cuda).")
    ap.add_argument("--reranker_batch_size", type=int, default=None, help="Override reranker batch size.")
    args = ap.parse_args()

    config = load_config(args.config)

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
    vector_cfg = config.get("vector", {}) if isinstance(config.get("vector"), dict) else {}
    rerank_cfg = config.get("reranker", {}) if isinstance(config.get("reranker"), dict) else {}
    fallback_cfg = config.get("fallback", {}) if isinstance(config.get("fallback"), dict) else {}

    stage1_cfg_val = vector_cfg.get("top_k_stage1") if isinstance(vector_cfg, dict) else None
    stage1_topk = stage1_cfg_val if stage1_cfg_val is not None else 24
    if args.stage1_topk is not None:
        stage1_topk = args.stage1_topk
    elif args.mmr_pool is not None:
        stage1_topk = args.mmr_pool
    stage1_topk = max(int(stage1_topk), 1)

    stage2_cfg_val = rerank_cfg.get("top_k_stage2") if isinstance(rerank_cfg, dict) else None
    stage2_topk = stage2_cfg_val if stage2_cfg_val is not None else 6
    if args.k is not None:
        stage2_topk = args.k
    stage2_topk = int(stage2_topk)
    if stage2_topk < 0:
        stage2_topk = 0

    fallback_lambda_cfg = fallback_cfg.get("lambda") if isinstance(fallback_cfg, dict) else None
    fallback_lambda = fallback_lambda_cfg if fallback_lambda_cfg is not None else 0.6
    if args.mmr_lambda is not None:
        fallback_lambda = args.mmr_lambda
    fallback_lambda = float(fallback_lambda)

    fallback_strategy_val = fallback_cfg.get("strategy") if isinstance(fallback_cfg, dict) else None
    fallback_strategy = str(fallback_strategy_val or "mmr")

    reranker_name_val = rerank_cfg.get("model") if isinstance(rerank_cfg, dict) else None
    reranker_name = reranker_name_val if reranker_name_val is not None else "none"
    if args.reranker is not None:
        reranker_name = args.reranker

    reranker_device = args.reranker_device or rerank_cfg.get("device")
    reranker_batch_size = args.reranker_batch_size or rerank_cfg.get("batch_size")
    reranker_batch_size = int(reranker_batch_size) if reranker_batch_size not in (None, "") else None

    stage1_topk = max(stage1_topk, stage2_topk if stage2_topk > 0 else stage1_topk, 1)

    reranker = get_reranker(reranker_name, device=reranker_device, batch_size=reranker_batch_size)

    skip_reasons = []
    if stage2_topk <= 0:
        skip_reasons.append("top_k_stage2<=0")
    if not reranker.available():
        skip_reasons.append(reranker.status() if hasattr(reranker, "status") else "unavailable")
    if skip_reasons:
        joined = "; ".join(skip_reasons)
        print(f"[retrieval] reranker skipped ({joined}); fallback={fallback_strategy}")

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
            stage1_topk=stage1_topk,
            stage2_topk=stage2_topk,
            fallback_strategy=fallback_strategy,
            fallback_lambda=fallback_lambda,
            plan_doc_class=plan.get("doc_class")
        )
        (out_dir / f"{tid}.json").write_text(json.dumps(bundle, indent=2), encoding="utf-8")
        print(f"[retrieval] wrote {out_dir / (tid + '.json')}")

if __name__ == "__main__":
    main()
