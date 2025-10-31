#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optional cross-encoder rerankers for DCL.

- Prefers BAAI/bge-reranker-v2-m3 (multilingual, strong & efficient).
- Falls back to no-op reranker if deps/models are unavailable.
- Deterministic: no sampling; pure forward scoring.

Refs:
- BGE reranker model list & usage (FlagEmbedding / Transformers).  # see docs
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import os
import math
import logging

logger = logging.getLogger(__name__)


@dataclass
class ScoredText:
    text: str
    score: float
    payload: dict


class BaseReranker:
    def available(self) -> bool:
        return False

    def rerank(self, query: str, candidates: List[dict], top_k: int) -> List[ScoredText]:
        """Return top_k candidates sorted by descending score."""
        # No-op: keep incoming order, uniform scores
        out = []
        for i, c in enumerate(candidates[:top_k]):
            out.append(ScoredText(text=c.get("text", ""), score=1.0, payload=c))
        return out


class BGEReranker(BaseReranker):
    """
    Uses a cross-encoder (BAAI/bge-reranker-*) via Transformers.
    """
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device
        self._ok = False
        self._load()

    def available(self) -> bool:
        return self._ok

    def _load(self):
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            self.model.eval()
            self.torch = torch
            self._ok = True
            logger.info("[reranker] loaded %s on %s", self.model_name, self.device)
        except Exception as e:
            logger.warning("[reranker] could not load %s: %s", self.model_name, e)
            self._ok = False

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + math.exp(-x))

    def rerank(self, query: str, candidates: List[dict], top_k: int) -> List[ScoredText]:
        if not self._ok or not candidates:
            return super().rerank(query, candidates, top_k)

        pairs = [(query, c.get("text", "")) for c in candidates]
        enc = self.tokenizer(
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with self.torch.no_grad():
            logits = self.model(**enc).logits.squeeze(-1).float().cpu().tolist()

        scored = []
        for logit, cand in zip(logits, candidates):
            scored.append(ScoredText(text=cand.get("text", ""), score=float(self._sigmoid(logit)), payload=cand))
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k]


def get_reranker(name_or_none: str, device: Optional[str] = None) -> BaseReranker:
    """
    name_or_none:
      - "none" or "" -> BaseReranker (no-op)
      - otherwise -> try BGEReranker(name_or_none)
    """
    if not name_or_none or name_or_none.lower() == "none":
        return BaseReranker()
    rr = BGEReranker(name_or_none, device=device)
    return rr if rr.available() else BaseReranker()
