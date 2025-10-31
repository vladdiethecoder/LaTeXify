#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import math
import shutil
import traceback
from pathlib import Path
from typing import List, Optional, Dict, Any

try:
    from llama_cpp import Llama
except Exception as e:
    print("[llamacpp-backend][FATAL] llama_cpp import failed:", e, file=sys.stderr)
    raise

def _parse_tensor_split(arg: Optional[str]) -> Optional[List[float]]:
    """
    Accepts:
      - None  -> None
      - "auto" -> even split across visible GPUs
      - "0.5,0.5" -> [0.5, 0.5]
    """
    if not arg:
        return None
    if isinstance(arg, list):
        return [float(x) for x in arg]
    s = str(arg).strip()
    if s == "auto":
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not cvd:
            return None
        gpus = [x for x in cvd.split(",") if x.strip() != ""]
        if not gpus:
            return None
        n = len(gpus)
        return [1.0 / n] * n
    try:
        return [float(x) for x in s.split(",") if x]
    except Exception:
        return None

def _file_ok(p: Path) -> bool:
    try:
        return p.exists() and p.is_file() and os.access(str(p), os.R_OK)
    except Exception:
        return False

class LlamaCppBackend:
    """
    A minimal, reliable llama.cpp wrapper with:
      - verbose init (so load failures are obvious)
      - safe defaults for Mixtral/CodeLlama GGUF
      - simple prompt builder (works with most *Instruct* variants)
    """
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        tensor_split: Optional[str] = None,
        seed: int = 12345,
        n_threads: Optional[int] = None,
        verbose_llama: bool = True,
        use_mmap: bool = True,
        use_mlock: bool = False,
    ):
        self.model_path = Path(model_path)
        if not _file_ok(self.model_path):
            raise FileNotFoundError(f"[llamacpp-backend] Model not readable: {self.model_path}")

        ts = _parse_tensor_split(tensor_split)
        if n_threads is None:
            try:
                import multiprocessing as mp
                n_threads = max(2, (mp.cpu_count() or 8) // 2)
            except Exception:
                n_threads = 8

        kwargs: Dict[str, Any] = dict(
            model_path=str(self.model_path),
            n_ctx=n_ctx,
            seed=seed,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            use_mmap=use_mmap,
            use_mlock=use_mlock,
            vocab_only=False,
            verbose=verbose_llama,
        )
        if ts is not None:
            kwargs["tensor_split"] = ts

        print("[llamacpp-backend] llama.init kwargs =", json.dumps({
            k: (v if k != "tensor_split" else f"{v} (auto={tensor_split=='auto'})")
            for k, v in kwargs.items()
        }, indent=2))

        try:
            self.llm = Llama(**kwargs)
        except Exception as e:
            # Surface the real C++ stderr diagnostics, if any.
            print("[llamacpp-backend][FATAL] Failed to initialize Llama with:", self.model_path, file=sys.stderr)
            traceback.print_exc()
            raise

    @staticmethod
    def _build_prompt(system: str, user: str) -> str:
        """
        A robust prompt that works for many *Instruct* GGUFs (LLaMA-2 style [INST]).
        If you later need exact templates, teach the backend via plan/metadata.
        """
        sys_block = (system or "").strip()
        usr_block = (user or "").strip()
        if sys_block:
            return f"<s>[INST] <<SYS>>\n{sys_block}\n<</SYS>>\n\n{usr_block} [/INST]"
        else:
            return f"<s>[INST] {usr_block} [/INST]"

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        top_p: float = 0.95,
        max_tokens: int = 512,
        repeat_penalty: float = 1.1,
    ) -> str:
        prompt = self._build_prompt(system_prompt, user_prompt)

        out = self.llm(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            stop=["</s>", "[/INST]"],
        )
        # llama.cpp returns a dict with 'choices' -> [ { 'text': ... } ]
        txt = out["choices"][0]["text"]
        return txt.strip()
