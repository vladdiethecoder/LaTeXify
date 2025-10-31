from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
from typing import Dict, List, Any

PREFER = ["nanonets-ocr2-3b", "qwen2-vl-ocr-2b", "nanonets-ocr-s"]

def pick_backend_text(ocr_outputs: Dict[str,str], prefer=PREFER) -> str:
    for be in prefer:
        t = ocr_outputs.get(be)
        if t: return t
    for t in (ocr_outputs or {}).values():
        if t: return t
    return ""

def normalize_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    s = re.sub(r"<\|im_.*?\|>", "", s).strip()  # strip chat artifacts
    return s

def chunk_from_blocks(blocks_jsonl: Path, out_path: Path) -> Dict[str, int]:
    blocks: List[dict] = []
    with blocks_jsonl.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip():
                blocks.append(json.loads(line))

    chunks: List[Dict[str, Any]] = []
    i = 0
    while i < len(blocks):
        b = blocks[i]
        btype = (b.get("block_type") or "Text").lower()
        text = b.get("latex_consensus") if btype in {"formula","equation","math","displaymath"} else pick_backend_text(b.get("ocr_outputs", {}))
        text = normalize_text(text or "")
        flags = (b.get("flag_reasons") or [])
        meta = {
            "block_id": b.get("block_id"),
            "page": b.get("page"),
            "block_type": b.get("block_type"),
            "bbox": b.get("bbox"),
            "flags": flags,
            "agreement_score": b.get("agreement_score"),
            "latex_agreement_score": b.get("latex_agreement_score"),
        }
        # tiny header→text merge for short following paragraph
        if btype == "header" and i+1 < len(blocks):
            nxt = blocks[i+1]
            if isinstance(nxt, dict) and (nxt.get("block_type","").lower() == "text"):
                nxt_text = normalize_text(pick_backend_text(nxt.get("ocr_outputs", {})))
                if 0 < len(nxt_text) <= 140:
                    text = (text + "\n\n" + nxt_text).strip()
                    if nxt.get("flag_reasons"):
                        meta["flags"] = list(set((meta["flags"] or []) + nxt["flag_reasons"]))
                    i += 1  # consume the next block

        chunks.append({
            "id": f"{b.get('page','page')}/{b.get('block_id','blk')}",
            "page": b.get("page"),
            "text": text,
            "metadata": meta,
        })
        i += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")
    return {"blocks_in": len(blocks), "chunks_out": len(chunks)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--blocks", default=None, help="defaults to <run_dir>/blocks_refined.jsonl")
    ap.add_argument("--out", default=None, help="defaults to <run_dir>/chunks.jsonl")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    blocks = Path(args.blocks) if args.blocks else (run_dir / "blocks_refined.jsonl")
    out = Path(args.out) if args.out else (run_dir / "chunks.jsonl")
    if not blocks.exists():
        print(f"Missing {blocks}", file=sys.stderr); sys.exit(2)
    stats = chunk_from_blocks(blocks, out)
    print(json.dumps({"run_dir": str(run_dir), "chunks": str(out), **stats}, indent=2))

if __name__ == "__main__":
    main()
