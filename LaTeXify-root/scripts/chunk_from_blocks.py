from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from chunk_strategies import build_chunks_for_mode, load_blocks

PREFER = ["nanonets-ocr2-3b", "qwen2-vl-ocr-2b", "nanonets-ocr-s"]


def pick_backend_text(ocr_outputs: Dict[str, str], prefer: Iterable[str] = PREFER) -> Tuple[str, Optional[str]]:
    for be in prefer:
        t = (ocr_outputs or {}).get(be)
        if isinstance(t, str) and t.strip():
            return t, be
    for be, t in (ocr_outputs or {}).items():
        if isinstance(t, str) and t.strip():
            return t, be
    return "", None


def normalize_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    s = re.sub(r"<\|im_.*?\|>", "", s).strip()  # strip chat artifacts
    return s


def _load_blocks(path: Path) -> List[dict]:
    blocks: List[dict] = []
    if not path.exists():
        return blocks
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                blocks.append(obj)
    return blocks


def _flags_from_block(block: dict) -> Dict[str, bool]:
    flags = block.get("flag_reasons") or block.get("flags") or {}
    if isinstance(flags, dict):
        return {str(k): bool(v) for k, v in flags.items()}
    if isinstance(flags, list):
        return {str(k): True for k in flags}
    if isinstance(flags, str):
        return {flags: True}
    return {}


def _merge_metadata(block: dict, text_backend: Optional[str]) -> Dict[str, Any]:
    return {
        "block_id": block.get("block_id"),
        "page": block.get("page"),
        "block_type": block.get("block_type") or block.get("label"),
        "bbox": block.get("bbox"),
        "flags": _flags_from_block(block),
        "agreement_score": block.get("agreement_score"),
        "latex_agreement_score": block.get("latex_agreement_score"),
        "source_backend": block.get("source_backend") or text_backend,
    }


def _prepare_blocks_for_mode(blocks: List[dict], prefer_backends: Iterable[str]) -> None:
    for block in blocks:
        block.setdefault("block_type", block.get("label"))
        if isinstance(block.get("text"), str) and block["text"].strip():
            continue
        if isinstance(block.get("latex_consensus"), str) and block["latex_consensus"].strip():
            block["text"] = normalize_text(block["latex_consensus"])
            block.setdefault("source_backend", "latex_consensus")
            continue
        text, backend = pick_backend_text(block.get("ocr_outputs", {}), prefer_backends)
        block["text"] = normalize_text(text)
        if backend:
            block.setdefault("source_backend", backend)


def _simple_chunks(blocks: List[dict], prefer_backends: Iterable[str]) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    i = 0
    while i < len(blocks):
        b = blocks[i]
        btype = (b.get("block_type") or b.get("label") or "Text").lower()
        if btype in {"formula", "equation", "math", "displaymath"}:
            text = b.get("latex_consensus") or ""
            backend = "latex_consensus"
        else:
            text, backend = pick_backend_text(b.get("ocr_outputs", {}), prefer_backends)
        text = normalize_text(text)

        meta = _merge_metadata(b, backend)

        # tiny header→text merge for short following paragraph
        if btype == "header" and i + 1 < len(blocks):
            nxt = blocks[i + 1]
            if isinstance(nxt, dict) and (nxt.get("block_type", "").lower() == "text"):
                nxt_text, nxt_backend = pick_backend_text(nxt.get("ocr_outputs", {}), prefer_backends)
                nxt_text = normalize_text(nxt_text)
                if 0 < len(nxt_text) <= 140:
                    text = (text + "\n\n" + nxt_text).strip()
                    nxt_flags = _flags_from_block(nxt)
                    if nxt_flags:
                        merged = {**(meta.get("flags") or {}), **nxt_flags}
                        meta["flags"] = merged
                    if nxt_backend and nxt_backend != meta.get("source_backend"):
                        meta.setdefault("merged_backends", []).append(nxt_backend)
                    i += 1  # consume the next block

        chunk_id = f"{b.get('page', 'page')}/{b.get('block_id', 'blk')}"
        chunks.append(
            {
                "id": chunk_id,
                "page": b.get("page"),
                "text": text,
                "bbox": b.get("bbox"),
                "block_type": b.get("block_type") or b.get("label"),
                "semantic_id": None,
                "source_backend": meta.get("source_backend"),
                "flags": meta.get("flags", {}),
                "metadata": meta,
            }
        )
        i += 1
    return chunks


def chunk_from_blocks(
    run_dir: Path,
    blocks_jsonl: Path,
    out_path: Path,
    mode: Optional[str] = None,
    prefer_backends: Iterable[str] = PREFER,
    max_chars: int = 1100,
    overlap: int = 150,
    min_par_len: int = 40,
) -> Dict[str, Any]:
    prefer_backends = list(prefer_backends)
    if mode:
        blocks = _load_blocks(blocks_jsonl) if blocks_jsonl.exists() else []
        if not blocks:
            blocks = load_blocks(str(run_dir), prefer_backends=list(prefer_backends))
        _prepare_blocks_for_mode(blocks, prefer_backends)
        chunks = build_chunks_for_mode(blocks, mode, max_chars, overlap, min_par_len)
    else:
        blocks = _load_blocks(blocks_jsonl)
        chunks = _simple_chunks(blocks, prefer_backends)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")

    return {
        "blocks_in": len(blocks),
        "chunks_out": len(chunks),
        "mode": mode or "page",
        "out": str(out_path),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--blocks", default=None, help="defaults to <run_dir>/blocks_refined.jsonl")
    ap.add_argument("--out", default=None, help="defaults to <run_dir>/<prefix>.chunks.jsonl or chunks.jsonl")
    ap.add_argument("--prefix", default=None, help="Prefix for chunk filename (e.g., user → user.chunks.jsonl)")
    ap.add_argument("--mode", choices=["user", "assessment", "rubric", "assignment"], default=None,
                    help="Semantic chunking strategy to apply")
    ap.add_argument("--max_chars", type=int, default=1100)
    ap.add_argument("--overlap", type=int, default=150)
    ap.add_argument("--min_par_len", type=int, default=40)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    blocks = Path(args.blocks) if args.blocks else (run_dir / "blocks_refined.jsonl")

    prefix = args.prefix or args.mode
    if args.out:
        out = Path(args.out)
    elif prefix:
        out = run_dir / f"{prefix}.chunks.jsonl"
    else:
        out = run_dir / "chunks.jsonl"

    if not blocks.exists() and not args.mode:
        print(f"Missing {blocks}", file=sys.stderr)
        sys.exit(2)

    stats = chunk_from_blocks(
        run_dir,
        blocks,
        out,
        mode=args.mode,
        max_chars=args.max_chars,
        overlap=args.overlap,
        min_par_len=args.min_par_len,
    )
    print(json.dumps({"run_dir": str(run_dir), **stats}, indent=2))


if __name__ == "__main__":
    main()
