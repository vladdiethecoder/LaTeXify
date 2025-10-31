#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model registry / discovery for local GGUFs.
"""
from __future__ import annotations
import os
from typing import Optional
from synth_backends.llamacpp_backend import discover_gguf

def pick_local_model(gguf_path: Optional[str], hf_cache: Optional[str]) -> str:
    if gguf_path and os.path.isfile(gguf_path):
        return gguf_path
    cand = discover_gguf(hf_cache)
    if not cand:
        raise SystemExit(
            "No .gguf model found. Pass --gguf-model /path/to/model.gguf "
            "or set HF_HOME / --hf-cache to your huggingface cache."
        )
    return cand
