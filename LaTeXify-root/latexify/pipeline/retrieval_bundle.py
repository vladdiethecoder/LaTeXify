# latexify/pipeline/retrieval_bundle.py
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Hard deps only when actually retrieving
try:
    import faiss  # type: ignore
    _FAISS_OK = True
except Exception:
    _FAISS_OK = False

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _SBERT_OK = True
except Exception:
    _SBERT_OK = False


SEED = 42  # determinism for any sorting / selection

###############################################################################
# Data model
###############################################################################

@dataclass
class Chunk:
    id: str
    text: str
    page: Optional[int]
    label: Optional[str]
    source_image: Optional[str]
    score: float
    source_name: str  # which index (assignment / assessment / rubric / user)

@dataclass
class UserAnswer:
    chunks: List[Chunk]
    flags: Dict[str, bool]

@dataclass
class ContextBundle:
    task_id: str
    question: str
    rubric: List[Chunk]
    assignment_rules: List[Chunk]
    assessment: List[Chunk]
    user_answer: UserAnswer


###############################################################################
# Helpers
###############################################################################

def _read_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                # tolerate a bad line and keep going
                pass
    return rows


def _load_meta(run_dir: Path) -> Dict:
    meta_p = run_dir / "faiss.meta.json"
    return json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}


def _load_corpus(run_dir: Path) -> Dict[str, Dict]:
    """
    Build id->record where record includes text + minimal fields we need.
    Priority:
      1) chunks/*.jsonl
      2) chunks.jsonl
      3) fallback from latex_docs.jsonl (compose text from title/Q/A/code)
    """
    # 1) chunks/*.jsonl
    for p in sorted((run_dir / "chunks").glob("*.jsonl")):
        rows = _read_jsonl(p)
        if rows:
            return {str(r.get("id")): r for r in rows if r.get("id") and r.get("text")}

    # 2) chunks.jsonl
    rows = _read_jsonl(run_dir / "chunks.jsonl")
    if rows:
        return {str(r.get("id")): r for r in rows if r.get("id") and r.get("text")}

    # 3) docs fallback
    docs = _read_jsonl(run_dir / "latex_docs.jsonl")
    out: Dict[str, Dict] = {}
    for rec in docs:
        cid = rec.get("id")
        if not cid:
            continue
        title = (rec.get("title") or "").strip()
        question = (rec.get("question") or "").strip()
        answer = (rec.get("answer") or "").strip()
        code = rec.get("code_blocks") or []
        code_excerpt = "\n\n".join(code[:2]).strip()
        text = "\n\n".join([t for t in [title, question, answer, code_excerpt] if t]).strip()
        if text:
            out[str(cid)] = {
                "id": cid,
                "text": text,
                "page": 1,
                "label": "kb",
                "source_image": rec.get("url"),
                "ocr_model": rec.get("source"),
                "bbox": [0, 0, 0, 0],
            }
    return out


def _embed_query(q: str, model_name: str) -> Tuple[List[float], "SentenceTransformer"]:
    if not _SBERT_OK:
        raise RuntimeError("sentence-transformers not installed in this environment.")
    model = SentenceTransformer(model_name)
    vec = model.encode([q], convert_to_numpy=True, normalize_embeddings=True)[0].astype("float32")
    return vec, model


def _search(run_dir: Path, query: str, top_k: int, source_name: str) -> List[Tuple[str, float]]:
    """
    Returns list of (id, score) sorted by score desc, id asc (deterministic on ties).
    If FAISS/SBERT not available or index missing, returns empty list.
    """
    if not (_FAISS_OK and _SBERT_OK):
        return []

    idx_p = run_dir / "faiss.index"
    meta = _load_meta(run_dir)
    if not idx_p.exists() or not meta:
        return []

    model_name = meta.get("model") or "sentence-transformers/all-MiniLM-L6-v2"
    qvec, _ = _embed_query(query, model_name)
    index = faiss.read_index(str(idx_p))
    D, I = index.search(qvec.reshape(1, -1), max(1, top_k * 5))  # over-fetch; we MMR later
    scores = D[0].tolist()
    # ids in meta["ids"] are aligned with order in the index build
    ids_meta: List[str] = meta.get("ids") or []
    hits: List[Tuple[str, float]] = []
    for i, s in zip(I[0].tolist(), scores):
        if 0 <= i < len(ids_meta):
            hits.append((str(ids_meta[i]), float(s)))

    # stable sort by (-score, id)
    hits.sort(key=lambda t: (-t[1], t[0]))
    return hits[: max(1, top_k * 5)]


def _mmr_rerank(
    query_vec: List[float],
    candidate_ids: List[str],
    corpus: Dict[str, Dict],
    model: "SentenceTransformer",
    k: int,
    lambda_: float = 0.7,
) -> List[str]:
    """
    Simple MMR on candidate texts. Deterministic stable order on ties.
    """
    import numpy as np

    texts = [corpus[cid]["text"] for cid in candidate_ids if cid in corpus]
    if not texts:
        return candidate_ids[:k]

    embs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    q = np.array(query_vec, dtype="float32")
    # cosine via dot (vectors normalized)
    sim_to_query = (embs @ q)

    selected: List[int] = []
    remaining = list(range(len(texts)))

    def pairwise_max(idx: int, chosen: List[int]) -> float:
        if not chosen:
            return 0.0
        # max similarity to any already selected
        return float(np.max(embs[idx] @ embs[chosen].T))

    for _ in range(min(k, len(texts))):
        best_i = None
        best_score = -1e9
        for i in remaining:
            diversity = pairwise_max(i, selected)
            score = lambda_ * float(sim_to_query[i]) - (1 - lambda_) * diversity
            if (score > best_score) or (math.isclose(score, best_score) and (best_i is None or i < best_i)):
                best_score = score
                best_i = i
        selected.append(best_i)  # type: ignore
        remaining.remove(best_i)  # type: ignore

    final_ids = [candidate_ids[i] for i in selected]
    return final_ids


def _assemble_chunks(
    hits: List[Tuple[str, float]],
    corpus: Dict[str, Dict],
    source_name: str,
    model_name: str,
    question: str,
    top_k: int,
    evidence_log: Optional[Path] = None,
    source_tag: Optional[str] = None,
) -> List[Chunk]:
    # Fallback: if we have no hits but do have a corpus, take the deterministic head
    if not hits and corpus:
        if evidence_log is not None:
            with evidence_log.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"event": "fallback_corpus_head", "source": source_tag or source_name, "k": top_k}) + "\n")
        # stable order by id; take first k
        ids = sorted(corpus.keys())[:top_k]
        out: List[Chunk] = []
        for cid in ids:
            r = corpus[cid]
            out.append(
                Chunk(
                    id=str(cid),
                    text=str(r.get("text") or ""),
                    page=r.get("page"),
                    label=r.get("label"),
                    source_image=r.get("source_image"),
                    score=0.0,
                    source_name=source_name,
                )
            )
        out.sort(key=lambda c: (c.id,))  # deterministic
        return out

    if not hits:
        return []

    # Normal MMR path (requires SBERT)
    if not _SBERT_OK:
        # Should not happen when hits exist, but guard anyway
        ids = [cid for cid, _ in hits][:top_k]
        out: List[Chunk] = []
        for cid in ids:
            if cid in corpus:
                r = corpus[cid]
                out.append(
                    Chunk(
                        id=str(cid),
                        text=str(r.get("text") or ""),
                        page=r.get("page"),
                        label=r.get("label"),
                        source_image=r.get("source_image"),
                        score=0.0,
                        source_name=source_name,
                    )
                )
        out.sort(key=lambda c: (c.id,))
        return out

    # SBERT present → MMR
    qvec, model = _embed_query(question, model_name)
    cands = [cid for cid, _ in hits if cid in corpus]
    if not cands:
        return []
    reranked_ids = _mmr_rerank(qvec, cands, corpus, model, k=top_k)
    score_map = {cid: s for cid, s in hits}
    chosen: List[Chunk] = []
    for cid in reranked_ids[:top_k]:
        r = corpus[cid]
        chosen.append(
            Chunk(
                id=str(cid),
                text=str(r.get("text") or ""),
                page=r.get("page"),
                label=r.get("label"),
                source_image=r.get("source_image"),
                score=float(score_map.get(cid, 0.0)),
                source_name=source_name,
            )
        )
    chosen.sort(key=lambda c: (-c.score, c.id))
    return chosen


def build_context_bundle(
    task: Dict,
    indices: Dict[str, Path],
    k_user: int = 6,
    k_rubric: int = 6,
    k_assignment: int = 6,
    k_assessment: int = 6,
    evidence_dir: Path = Path("evidence"),
) -> ContextBundle:
    """
    Build a retrieval bundle for one task. Nonexistent indexes are tolerated.
    indices keys: {"assignment","assessment","rubric","user"} → each is a directory containing *.index + *.meta.json
    """
    task_id = str(task.get("id") or "T??")
    title = str(task.get("title") or task.get("anchor") or "Task")
    question = f"{task_id}: {title}"

    evidence_dir.mkdir(parents=True, exist_ok=True)
    ev = evidence_dir / f"{task_id}.json"
    def log(event: str, **details):
        rec = {"event": event, "task_id": task_id, **details}
        with ev.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def do_source(name: str, top_k: int) -> List[Chunk]:
        run_dir = indices.get(name)
        if not run_dir:
            log("skip_source", source=name, reason="missing_dir")
            return []
        run_dir = Path(run_dir)
        meta = _load_meta(run_dir)
        corpus = _load_corpus(run_dir)
        model_name = meta.get("model") or "sentence-transformers/all-MiniLM-L6-v2"
        hits = _search(run_dir, question, top_k, name)
        if not hits and corpus:
            log("retrieval_empty", source=name, reason="no_hits_or_deps")
        else:
            log("retrieval_raw", source=name, top_k=len(hits), model=model_name)
        chunks = _assemble_chunks(
            hits, corpus, name, model_name, question, top_k,
            evidence_log=ev, source_tag=name
        )
        log("retrieval_done", source=name, kept=len(chunks))
        return chunks

    rubric = do_source("rubric", k_rubric)
    assignment_rules = do_source("assignment", k_assignment)
    assessment = do_source("assessment", k_assessment)
    user_chunks = do_source("user", k_user)

    bundle = ContextBundle(
        task_id=task_id,
        question=question,
        rubric=rubric,
        assignment_rules=assignment_rules,
        assessment=assessment,
        user_answer=UserAnswer(chunks=user_chunks, flags={"ocr_uncertain": False}),
    )
    log("bundle_done", rubric=len(rubric), assignment=len(assignment_rules), assessment=len(assessment), user=len(user_chunks))
    return bundle


###############################################################################
# CLI for smoke use
###############################################################################

def _load_plan(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))

def main():
    ap = argparse.ArgumentParser(description="Build Retrieval Context Bundle for a task.")
    ap.add_argument("--plan", type=Path, default=Path("plan.json"))
    ap.add_argument("--task_id", type=str, default="T03", help="Which task to bundle (default: T03)")
    ap.add_argument("--assignment", type=Path, default=Path("kb/latex"))
    ap.add_argument("--assessment", type=Path, default=Path("kb/latex"))
    ap.add_argument("--rubric", type=Path, default=Path("kb/latex"))
    ap.add_argument("--user", type=Path, default=Path("kb/latex"))
    ap.add_argument("--k_user", type=int, default=6)
    ap.add_argument("--k_rubric", type=int, default=6)
    ap.add_argument("--k_assignment", type=int, default=6)
    ap.add_argument("--k_assessment", type=int, default=6)
    ap.add_argument("--out", type=Path, default=None, help="Optional path to save the bundle JSON")
    args = ap.parse_args()

    plan = _load_plan(args.plan)
    tasks = {t["id"]: t for t in plan.get("tasks", [])}
    task = tasks.get(args.task_id)
    if not task:
        raise SystemExit(f"Task id {args.task_id} not found in plan.")

    indices = {
        "assignment": args.assignment,
        "assessment": args.assessment,
        "rubric": args.rubric,
        "user": args.user,
    }

    bundle = build_context_bundle(
        task=task,
        indices=indices,
        k_user=args.k_user,
        k_rubric=args.k_rubric,
        k_assignment=args.k_assignment,
        k_assessment=args.k_assessment,
        evidence_dir=Path("evidence"),
    )

    out_obj = {
        "task_id": bundle.task_id,
        "question": bundle.question,
        "rubric": [asdict(c) for c in bundle.rubric],
        "assignment_rules": [asdict(c) for c in bundle.assignment_rules],
        "assessment": [asdict(c) for c in bundle.assessment],
        "user_answer": {
            "chunks": [asdict(c) for c in bundle.user_answer.chunks],
            "flags": bundle.user_answer.flags,
        },
    }

    js = json.dumps(out_obj, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(js, encoding="utf-8")
        print(f"Wrote {args.out}")
    else:
        print(js)


if __name__ == "__main__":
    main()
