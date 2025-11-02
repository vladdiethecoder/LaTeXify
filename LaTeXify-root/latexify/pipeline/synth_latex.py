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
from typing import Dict, List, Optional, Tuple

import argparse
import json
import os
import sys
import textwrap

# Use relative imports for pipeline modules
from .model_backends import LlamaCppBackend, LlamaCppConfig
from .model_router import choose_model_for_text, RouteDecision
from . import synth_text
from .synth_shared import capabilities_from_text, read_bundle
from .specialist_router import select_specialist


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

SYSTEM_INSTR = textwrap.dedent(
    """You are a LaTeX synthesis agent. Produce ONLY a LaTeX snippet (no preamble, no \documentclass, no \begin{document}).
Respect LiX classes when relevant (lix_textbook or lix_article) and prefer semantic, clean LaTeX.
- Math: amsmath environments (align, equation) with labels; inline math as $...$.
- Tables: booktabs + tabular with sensible column alignment; no vertical lines; include \label.
- Figures: use \includegraphics[width=\linewidth]{<placeholder>} with \caption and \label.
- No boilerplate, no explanations, no markdown fences.
"""
)

PROMPT_TEMPLATE = textwrap.dedent(
    """{system}

# Source
{ocr_text}

# Context (RAG)
{rag_context}

# Task
Generate a high-quality LaTeX snippet for this section. Only the snippet body. If unsure about a value, use sensible placeholders and add a % TODO comment. Keep it compact and compilable.
"""
)


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


def _ensure_out(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# Backwards compatibility for tests and CLI helpers
def build_snippet(bundle: Dict) -> str:
    snippet, _ = synth_text.synthesize(bundle)
    return snippet


def synthesize_snippet(
    bundle_or_path: Dict | Path | str,
    out_dir: Optional[Path] = None,
    *,
    kb_dir: Optional[Path] = None,
) -> str | Path:
    """Generate a deterministic snippet. Accepts either bundle dict or path."""
    if isinstance(bundle_or_path, (str, Path)):
        bundle_path = Path(bundle_or_path)
        bundle = read_bundle(bundle_path)
        snippet = synth_text.synthesize_cli(bundle)
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{bundle.get('task_id', bundle_path.stem)}.tex"
            out_path.write_text(snippet, encoding="utf-8")
            return out_path
        return snippet
    bundle = bundle_or_path
    snippet_text, _ = synth_text.synthesize(bundle)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{bundle.get('task_id', 'snippet')}.tex"
        out_path.write_text(snippet_text, encoding="utf-8")
        return out_path
    return snippet_text


def _load_plan_metadata(path: Optional[Path]) -> Dict[str, Dict[str, str]]:
    if not path or not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    mapping: Dict[str, Dict[str, str]] = {}
    for task in data.get("tasks", []):
        if not isinstance(task, dict):
            continue
        tid = task.get("id")
        if not tid:
            continue
        info: Dict[str, str] = {}
        ctype = task.get("content_type")
        if ctype:
            info["content_type"] = str(ctype)
        kind = task.get("kind")
        if kind:
            info["kind"] = str(kind)
        asset_path = task.get("asset_path")
        if asset_path:
            info["asset_path"] = str(asset_path)
        task_type = task.get("type")
        if task_type:
            info["type"] = str(task_type)
        asset_source = task.get("asset_source_type")
        if asset_source:
            info["asset_source_type"] = str(asset_source)
        if task.get("asset_page_index") is not None:
            info["asset_page_index"] = str(task.get("asset_page_index"))
        asset_id = task.get("asset_id")
        if asset_id:
            info["asset_id"] = str(asset_id)
        if info:
            mapping[str(tid)] = info
    return mapping


def _build_prompt(ocr_text: str, rag_context: str) -> str:
    return PROMPT_TEMPLATE.format(
        system=SYSTEM_INSTR.strip(),
        ocr_text=ocr_text.strip(),
        rag_context=rag_context.strip(),
    )


def _load_backend_for(
    override_model: Optional[Path],
    route: RouteDecision,
    args: Args,
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
    plan_metadata: Dict[str, Dict[str, str]],
) -> Tuple[Path, Path]:
    b = read_bundle(bundle_path)
    bid = b.get("id") or bundle_path.stem.replace(".bundle", "")
    ocr = b.get("prompt") or b.get("ocr") or ""
    ctx = b.get("context") or ""
    body = _build_prompt(ocr, ctx)
    task_info = plan_metadata.get(bid, {})
    if task_info:
        for key in (
            "content_type",
            "asset_path",
            "asset_source_type",
            "asset_page_index",
            "asset_id",
            "kind",
        ):
            if key in task_info and key not in b:
                b[key] = task_info[key]
        if task_info.get("type") and "type" not in b:
            b["type"] = task_info["type"]
        if task_info.get("kind") and "kind" not in b:
            b["kind"] = task_info["kind"]
    decision = select_specialist(b, task_info)
    if decision:
        snippet, caps = decision.run(b)
        return _write_outputs(
            out_dir,
            bid,
            snippet,
            caps,
            route_reason=decision.reason,
            model_path=None,
            seed=args.seed,
            route_metadata=decision.metadata,
        )
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
        completion = completion.split("\n", 1)[-1].strip()
    # Write outputs
    return _write_outputs(
        out_dir,
        bid,
        completion,
        capabilities_from_text(completion),
        route_reason=route.reason,
        model_path=route.model_path,
        seed=args.seed,
        route_metadata={"prompt_version": "llm", "prompt_path": "", "specialist": "", "tags": ""},
    )


def _write_outputs(
    out_dir: Path,
    bid: str,
    text: str,
    capabilities: List[str],
    route_reason: str,
    model_path: Optional[Path],
    seed: int,
    route_metadata: Optional[Dict[str, str]] = None,
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
    if route_metadata:
        meta["route_metadata"] = route_metadata
    out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[synth] {bid} → {out_tex}  ({route_reason})")
    return out_tex, out_meta


def parse_args() -> Args:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundles-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--gguf-model",
        type=Path,
        default=None,
        help="Override model path (.gguf). If set, routing is skipped.",
    )
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
    plan_metadata = _load_plan_metadata(args.plan)
    bundle_files = sorted([
        p
        for p in args.bundles_dir.rglob("*.json")
        if p.suffix == ".json" and not p.name.endswith(".meta.json")
    ])
    if not bundle_files:
        print(f"[synth][WARN] No bundle JSON files found in {args.bundles_dir}")
        return 0
    for bf in bundle_files:
        try:
            synth_one_bundle(bf, args.out, backend_cache, args, plan_metadata)
        except Exception as e:
            print(f"[synth][ERR] {bf.name}: {e}", file=sys.stderr)
            continue
    return 0


if __name__ == "__main__":
    raise SystemExit(main())