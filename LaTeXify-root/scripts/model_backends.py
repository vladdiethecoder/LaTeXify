#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model_backends.py

Thin wrapper around llama-cpp-python to run local GGUF models with sane defaults:
- Multi-GPU: `tensor_split="auto"` will distribute across visible GPUs.
- Determinism: controlled by `seed` (pass -1 for non-deterministic).
- Context: `n_ctx` configurable (defaults 4096).

Requires: `pip install llama-cpp-python` (build with CUDA or install a CUDA wheel if desired).
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import os
import re

try:
    from llama_cpp import Llama
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "llama-cpp-python not available. Install it first (GPU example):\n"
        '  CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python\n'
        "or install a CUDA wheel per the project docs."
    ) from e


def _visible_gpus() -> List[int]:
    """
    Returns parsed CUDA_VISIBLE_DEVICES indices if set, otherwise tries to detect count via nvidia-smi.
    If nothing is detectable, returns [] meaning CPU-only or unknown.
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cvd:
        # e.g. "0,1" -> [0,1]; also allow "GPU-..." notations to count entries
        toks = [t for t in re.split(r"[,\s]+", cvd) if t]
        idxs = []
        for t in toks:
            try:
                idxs.append(int(t))
            except ValueError:
                # non-index token (UUID); just count it
                idxs.append(len(idxs))
        return idxs
    # Fallback: try to query nvidia-smi count
    try:
        import subprocess
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True)
        lines = [ln for ln in out.splitlines() if ln.strip()]
        return list(range(len(lines)))
    except Exception:
        return []


def _parse_tensor_split(spec: str | Sequence[float] | None) -> Optional[List[float]]:
    """
    Parse tensor_split argument:
    - "auto": even split across visible GPUs
    - "1,1,2": explicit ratios across N GPUs
    - None / []: return None to let backend default
    """
    if spec is None:
        return None
    if isinstance(spec, (list, tuple)):
        vals = [float(x) for x in spec]
        return vals if vals else None
    s = str(spec).strip().lower()
    if s == "auto":
        gpus = _visible_gpus()
        if len(gpus) <= 0:
            return None
        # Even split across N GPUs
        frac = 1.0 / float(len(gpus))
        return [frac] * len(gpus)
    # Comma-separated floats
    try:
        vals = [float(x) for x in re.split(r"[,\s]+", s) if x]
        return vals if vals else None
    except Exception:
        return None


@dataclass
class LlamaCppConfig:
    model_path: Path
    n_ctx: int = 4096
    n_batch: int = 512
    seed: int = 1337
    n_gpu_layers: int = -1           # -1 => offload as much as possible to GPU(s)
    tensor_split: str | Sequence[float] | None = "auto"  # "auto" -> split across visible GPUs
    n_threads: Optional[int] = None
    n_threads_batch: Optional[int] = None
    verbose: bool = True
    chat_format: Optional[str] = None  # auto-detect if None


class LlamaCppBackend:
    """
    Convenience wrapper. Keeps a single Llama instance per model_path.
    """

    def __init__(self, cfg: LlamaCppConfig):
        self.cfg = cfg
        self._llm = self._load()

    def _load(self) -> Llama:
        ts = _parse_tensor_split(self.cfg.tensor_split)
        llm = Llama(
            model_path=str(self.cfg.model_path),
            n_ctx=self.cfg.n_ctx,
            n_batch=self.cfg.n_batch,
            seed=self.cfg.seed,
            n_gpu_layers=self.cfg.n_gpu_layers,
            tensor_split=ts,
            n_threads=self.cfg.n_threads,
            n_threads_batch=self.cfg.n_threads_batch,
            chat_format=self.cfg.chat_format,
            verbose=self.cfg.verbose,
        )
        return llm

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.95,
        top_k: int = 40,
        stop: Optional[Iterable[str]] = None,
        repeat_penalty: float = 1.05,
    ) -> str:
        """
        Return raw text completion from llama.cpp high-level call.
        """
        out = self._llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=list(stop) if stop else None,
            repeat_penalty=repeat_penalty,
            echo=False,
        )
        # High-level API returns dict with 'choices'
        txt = out.get("choices", [{}])[0].get("text", "")
        return txt
