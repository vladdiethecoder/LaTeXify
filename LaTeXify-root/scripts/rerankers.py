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
from typing import List, Optional

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
    """No-op reranker that exposes availability diagnostics."""

    def __init__(self, name: str = "none", *, available: bool = False, status: str = "disabled") -> None:
        self.name = name
        self._available = available
        self._status = status

    def available(self) -> bool:
        return self._available

    def status(self) -> str:
        return self._status

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
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        *,
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
    ):
        super().__init__(name=model_name, available=False, status="loading")
        self.model_name = model_name
        self.device = device
        self.batch_size = int(batch_size) if batch_size else 8
        self._load()

    def available(self) -> bool:
        return self._available

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
            self._available = True
            self._status = f"loaded on {self.device}"
            logger.info("[reranker] loaded %s on %s", self.model_name, self.device)
        except Exception as e:
            logger.warning("[reranker] could not load %s: %s", self.model_name, e)
            self._available = False
            self._status = f"error: {e}"  # keep reason for diagnostics

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + math.exp(-x))

    def rerank(self, query: str, candidates: List[dict], top_k: int) -> List[ScoredText]:
        if not self.available() or not candidates:
            return super().rerank(query, candidates, top_k)

        pairs = [(query, c.get("text", "")) for c in candidates]
        logits: List[float] = []
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            enc = self.tokenizer(
                [p[0] for p in batch],
                [p[1] for p in batch],
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with self.torch.no_grad():
                outputs = self.model(**enc).logits.squeeze(-1)
                if outputs.ndim == 0:
                    batch_logits = [float(outputs)]
                else:
                    batch_logits = [float(x) for x in outputs.cpu().tolist()]
            logits.extend(batch_logits)

        scored = []
        for logit, cand in zip(logits, candidates):
            scored.append(
                ScoredText(text=cand.get("text", ""), score=float(self._sigmoid(logit)), payload=cand)
            )
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k]


def get_reranker(
    name_or_none: str,
    *,
    device: Optional[str] = None,
    batch_size: Optional[int] = None,
) -> BaseReranker:
    """
    name_or_none:
      - "none" or "" -> BaseReranker (no-op)
      - otherwise -> try BGEReranker(name_or_none)
    """
    if not name_or_none or name_or_none.lower() == "none":
        return BaseReranker(name="none", available=False, status="disabled via config")
    rr = BGEReranker(name_or_none, device=device, batch_size=batch_size)
    return rr
