#!/usr/bin/env python3
import argparse, sys
from scripts.synth_backends.llamacpp_backend import LlamaCppBackend

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--ctx", type=int, default=4096)
ap.add_argument("--n-gpu-layers", type=int, default=-1)
ap.add_argument("--tensor-split", default=None)
args = ap.parse_args()

backend = LlamaCppBackend(
    model_path=args.model,
    n_ctx=args.ctx,
    n_gpu_layers=args.n_gpu_layers,
    tensor_split=args.tensor_split,
)

out = backend.generate(
    system_prompt="You are a helpful assistant.",
    user_prompt="Say 'READY' and nothing else.",
    max_tokens=5,
    temperature=0.0,
)
print("MODEL SAYS:", out)
