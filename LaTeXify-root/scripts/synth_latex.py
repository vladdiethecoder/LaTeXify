#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
synth_latex.py

Synthesis Agent: generates LaTeX snippets from OCR/RAG bundles using local GGUF models.

Key features:
- llama-cpp backend (GPU/CPU). Multi-GPU via tensor_split="auto".
- Content-aware routing:
    * math-heavy → CodeLlama-70B-Instruct
    * tables/figures → Mixtral-8x7B-Instruct
    * prose/default → first available GGUF
- Deterministic when seed is provided.

Inputs:
- --bundles-dir: directory containing *.bundle.json (each holds OCR text + RAG context).
- --out: directory to write *.tex (and optional *.meta.json).
- Optional direct model override: --gguf-model /path/model.gguf
- Optional cache root: --hf-cache /path/to/hf/cache
- Generation controls: --ctx, --n-gpu-layers, --tensor-split, --seed, --max-tokens

Bundle schema (minimal expectation):
{
  "id": "Q1",
  "prompt": "... OCR text ...",
  "context": "RAG passages ...",
  "class": "lix_textbook" | "lix_article" | "article" (optional)
}

Output:
- build/snippets/<id>.tex
- build/snippets/<id>.meta.json  (capability tags for gate6_preamble)
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import argparse
import json
import os
import sys
import textwrap

from model_backends import LlamaCppBackend, LlamaCppConfig
from model_router import choose_model_for_text, RouteDecision
import synth_table
import synth_formula
import synth_figure


DEFAULT_CTX = 4096
DEFAULT_MAX_TOK = 700
STOP = ["\\end{document}", "\n\\section", "\n\\chapter"]

CAPABILITY_HINTS = [
    ("amsmath", ["\\begin{align", "\\begin{equation", "\\[", "\\("]),
    ("amssymb", ["\\mathbb{", "\\mathcal{", "\\mathfrak{"]),
    ("graphicx", ["\\includegraphics"]),
    ("booktabs", ["\\toprule", "\\midrule", "\\bottomrule"]),
    ("hyperref", ["\\url{", "\\href{"]),
]

SYSTEM_INSTR = """You are a LaTeX synthesis agent. Produce ONLY a LaTeX snippet (no preamble, no \\documentclass, no \\begin{document}).
Respect LiX classes when relevant (lix_textbook or lix_article) and prefer semantic, clean LaTeX.
- Math: amsmath environments (align, equation) with labels; inline math as $...$.
- Tables: booktabs + tabular with sensible column alignment; no vertical lines; include \\label.
- Figures: use \\includegraphics[width=\\linewidth]{<placeholder>} with \\caption and \\label.
- No boilerplate, no explanations, no markdown fences.
"""

PROMPT_TEMPLATE = """{system}

# Source
{ocr_text}

# Context (RAG)
{rag_context}

# Task
Generate a high-quality LaTeX snippet for this section. Only the snippet body. If unsure about a value, use sensible placeholders and add a % TODO comment. Keep it compact and compilable.
"""

@dataclass
class Args:
    bundles_dir: Path
    out: Path
    gguf_model: Optional[Path]
    hf_cache: Optional[Path]
    seed: int
    n_ctx: int
    n_gpu_layers: int
    tensor_split: str
    max_tokens: int
    n_threads: Optional[int]
    n_threads_batch: Optional[int]
    plan: Optional[Path]

def _read_json(p: Path) -> Dict:
    return json.loads(p.read_text(encoding="utf-8"))

def _ensure_out(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _capabilities_from_text(tex: str) -> List[str]:
    caps = []
    for name, needles in CAPABILITY_HINTS:
        if any(n in tex for n in needles):
            caps.append(name)
    return sorted(set(caps))


def _load_plan_types(path: Optional[Path]) -> Dict[str, str]:
    if not path or not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    mapping: Dict[str, str] = {}
    for task in data.get("tasks", []):
        tid = task.get("id")
        ctype = task.get("content_type")
        if tid and ctype:
            mapping[str(tid)] = str(ctype)
    return mapping


def _choose_specialist(content_type: Optional[str]):
    if not content_type:
        return None
    kind = content_type.lower()
    if "table" in kind:
        return synth_table.synthesize
    if any(token in kind for token in ("formula", "equation", "math")):
        return synth_formula.synthesize
    if any(token in kind for token in ("figure", "image", "picture")):
        return synth_figure.synthesize
    return None

def _build_prompt(ocr_text: str, rag_context: str) -> str:
    return PROMPT_TEMPLATE.format(
        system=SYSTEM_INSTR.strip(),
        ocr_text=ocr_text.strip(),
        rag_context=rag_context.strip()
    )

def _load_backend_for(
    override_model: Optional[Path],
    route: RouteDecision,
    args: Args
) -> LlamaCppBackend:
    model_path = override_model or route.model_path
    cfg = LlamaCppConfig(
        model_path=model_path,
        n_ctx=args.n_ctx,
        n_batch=512,
        seed=args.seed,
        n_gpu_layers=args.n_gpu_layers,
        tensor_split=args.tensor_split,
        n_threads=args.n_threads,
        n_threads_batch=args.n_threads_batch,
        verbose=True,
    )
    return LlamaCppBackend(cfg)

def synth_one_bundle(
    bundle_path: Path,
    out_dir: Path,
    backend_cache: Dict[str, LlamaCppBackend],
    args: Args,
    plan_types: Dict[str, str],
) -> Tuple[Path, Path]:
    b = _read_json(bundle_path)
    bid = b.get("id") or bundle_path.stem.replace(".bundle", "")
    ocr = b.get("prompt") or b.get("ocr") or ""
    ctx = b.get("context") or ""
    body = _build_prompt(ocr, ctx)

    specialist = _choose_specialist(plan_types.get(bid))
    if specialist:
        snippet, caps = specialist(b)
        return _write_outputs(out_dir, bid, snippet, caps, route_reason="specialist", model_path=None, seed=args.seed)

    # Routing (unless --gguf-model is explicitly provided)
    if args.gguf_model:
        route = RouteDecision(reason="override", model_path=args.gguf_model)
    else:
        route = choose_model_for_text(ocr + "\n" + ctx, args.hf_cache)

    key = str(route.model_path.resolve())
    if key not in backend_cache:
        backend_cache[key] = _load_backend_for(args.gguf_model, route, args)

    llm = backend_cache[key]
    completion = llm.generate(
        body,
        max_tokens=args.max_tokens,
        temperature=0.2,
        top_p=0.95,
        top_k=40,
        stop=STOP,
        repeat_penalty=1.05,
    ).strip()

    # Trim accidental fencing or markdown
    if completion.startswith("```"):
        completion = completion.strip("` \n")
        # try to drop 'latex' first token if present
        completion = completion.split("\n", 1)[-1].strip()

    # Write outputs
    out_tex = out_dir / f"{bid}.tex"
    out_meta = out_dir / f"{bid}.meta.json"
    return _write_outputs(
        out_dir,
        bid,
        completion,
        _capabilities_from_text(completion),
        route_reason=route.reason,
        model_path=route.model_path,
        seed=args.seed,
    )


def _write_outputs(
    out_dir: Path,
    bid: str,
    text: str,
    capabilities: List[str],
    route_reason: str,
    model_path: Optional[Path],
    seed: int,
) -> Tuple[Path, Path]:
    _ensure_out(out_dir)
    out_tex = out_dir / f"{bid}.tex"
    out_meta = out_dir / f"{bid}.meta.json"
    content = text + ("\n" if not text.endswith("\n") else "")
    out_tex.write_text(content, encoding="utf-8")
    meta = {
        "id": bid,
        "route": route_reason,
        "model_path": str(model_path) if model_path else None,
        "seed": seed,
        "capabilities": sorted(set(capabilities)),
    }
    out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[synth] {bid} → {out_tex}  ({route_reason})")
    return out_tex, out_meta

def parse_args() -> Args:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundles-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gguf-model", type=Path, default=None, help="Override model path (.gguf). If set, routing is skipped.")
    ap.add_argument("--hf-cache", type=Path, default=None, help="Optional HF cache root to include in model search.")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--ctx", type=int, default=DEFAULT_CTX)
    ap.add_argument("--plan", type=Path, help="Optional plan.json with content_type annotations")
    ap.add_argument("--n-gpu-layers", type=int, default=-1)
    ap.add_argument("--tensor-split", type=str, default="auto")
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOK)
    ap.add_argument("--n-threads", type=int, default=None)
    ap.add_argument("--n-threads-batch", type=int, default=None)
    ns = ap.parse_args()

    return Args(
        bundles_dir=ns.bundles_dir,
        out=ns.out,
        gguf_model=ns.gguf_model,
        hf_cache=ns.hf_cache,
        seed=ns.seed,
        n_ctx=ns.ctx,
        n_gpu_layers=ns.n_gpu_layers,
        tensor_split=ns.tensor_split,
        max_tokens=ns.max_tokens,
        n_threads=ns.n_threads,
        n_threads_batch=ns.n_threads_batch,
        plan=ns.plan,
    )

def main() -> int:
    args = parse_args()
    if not args.bundles_dir.exists():
        print(f"[synth][ERR] Bundles dir not found: {args.bundles_dir}", file=sys.stderr)
        return 2
    _ensure_out(args.out)

    backend_cache: Dict[str, LlamaCppBackend] = {}
    plan_types = _load_plan_types(args.plan)
    bundle_files = sorted(
        [
            p
            for p in args.bundles_dir.rglob("*.json")
            if p.suffix == ".json" and not p.name.endswith(".meta.json")
        ]
    )

    if not bundle_files:
        print(f"[synth][WARN] No bundle JSON files found in {args.bundles_dir}")
        return 0

    for bf in bundle_files:
        try:
            synth_one_bundle(bf, args.out, backend_cache, args, plan_types)
        except Exception as e:
            print(f"[synth][ERR] {bf.name}: {e}", file=sys.stderr)
            continue

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
