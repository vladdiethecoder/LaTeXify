#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DCL helper utilities:
- FAISS index loader with BGE-M3 query embedder
- Simple topK search
- Token/char budgeting helpers

Assumptions:
- build_index_bgem3.py wrote:
    <index_dir>/faiss.index
    <index_dir>/faiss.meta.json
  where meta["id_to_ref"] is a list of {id, text, ...} dicts.

Deterministic, pure-Python (Torch only for embedding).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Try FAISS (GPU not required)
try:
    import faiss  # type: ignore
except Exception as e:
    faiss = None
    logger.error("faiss not available: %s", e)


def _normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    return vecs / norms


class BGEM3QueryEncoder:
    """
    Minimal BGE-M3 query encoder.
    Uses FlagEmbedding if available; otherwise tries SentenceTransformer.
    """
    def __init__(self, device: Optional[str] = None):
        self.device = device
        self.impl = None
        self.kind = "none"
        self._load()

    def _load(self):
        # Prefer FlagEmbedding
        try:
            from FlagEmbedding import BGEM3FlagModel  # type: ignore
            self.impl = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True if self._has_cuda() else False, devices=self.device)
            self.kind = "flagembedding"
            logger.info("[embedder] using FlagEmbedding BGE-M3")
            return
        except Exception as e:
            logger.warning("[embedder] FlagEmbedding unavailable: %s", e)

        # Fallback: sentence-transformers (may be slower)
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self.impl = SentenceTransformer("BAAI/bge-m3", device=self.device or ("cuda" if self._has_cuda() else "cpu"))
            self.kind = "sentence-transformers"
            logger.info("[embedder] using SentenceTransformer BGE-M3")
            return
        except Exception as e:
            logger.warning("[embedder] SentenceTransformer unavailable: %s", e)

        self.kind = "none"
        self.impl = None
        logger.error("[embedder] No BGE-M3 embedder found. Queries will fail.")

    @staticmethod
    def _has_cuda() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    def encode_query(self, text: str) -> np.ndarray:
        if self.impl is None:
            raise RuntimeError("No embedder available (install FlagEmbedding or sentence-transformers).")
        if self.kind == "flagembedding":
            out = self.impl.encode(text, prompt_name="query")  # returns dict with 'dense_vecs'
            vec = np.asarray(out["dense_vecs"]).reshape(1, -1).astype("float32")
        else:
            # sentence-transformers returns (n, d)
            vec = np.asarray(self.impl.encode([text], normalize_embeddings=False)).astype("float32")
        return _normalize(vec)


@dataclass
class DocRef:
    id: str
    text: str
    meta: Dict[str, Any]


class FaissSearcher:
    def __init__(self, index_dir: str):
        index_dir = str(index_dir)
        self.dir = Path(index_dir)
        self.meta = self._read_meta(self.dir / "faiss.meta.json")
        self.dim = int(self.meta.get("dim", 1024))
        self.id_to_ref: List[DocRef] = []
        for rec in self.meta.get("id_to_ref", []):
            self.id_to_ref.append(
                DocRef(
                    id=str(rec.get("id", "")),
                    text=str(rec.get("text", "")),
                    meta=rec.get("meta", {}) if isinstance(rec.get("meta", {}), dict) else {},
                )
            )
        if faiss is None:
            raise RuntimeError("faiss is required to search")
        self.index = faiss.read_index(str(self.dir / "faiss.index"))
        self.encoder = BGEM3QueryEncoder()

    @staticmethod
    def _read_meta(path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Missing meta: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _take(self, ids: np.ndarray, scores: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for idx, sc in zip(ids[0][:top_k], scores[0][:top_k]):
            if idx < 0 or idx >= len(self.id_to_ref):
                continue
            ref = self.id_to_ref[idx]
            out.append({"id": ref.id, "text": ref.text, "score": float(sc), "meta": ref.meta})
        return out

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        q = self.encoder.encode_query(query).astype("float32")
        # Assume inner-product index on normalized embeddings
        scores, ids = self.index.search(q, top_k)
        return self._take(ids, scores, top_k)


# -------- Budget helpers --------

def est_tokens(s: str) -> int:
    # very rough: ~4 chars per token
    return max(1, int(len(s) / 4))


def cap_context(cands: List[str], max_tokens: int) -> List[str]:
    out, used = [], 0
    for c in cands:
        t = est_tokens(c)
        if used + t > max_tokens:
            break
        out.append(c)
        used += t
    return out
