#!/usr/bin/env python
import json, argparse, re, pathlib

p = argparse.ArgumentParser()
p.add_argument("--blocks", required=True)
p.add_argument("--out", required=True)
p.add_argument("--doc_id", default="")
args = p.parse_args()

doc_id = args.doc_id or None
out = pathlib.Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)

def guess_section_hint(t: str) -> str:
    # ultra-light heuristic for the dry-run
    t0 = (t or "").strip().lower()
    if t0.startswith("question "): return "Question"
    if "abstract" in t0: return "Abstract"
    if "introduction" in t0: return "Introduction"
    return ""

with open(args.blocks, "r", encoding="utf-8") as f, open(out, "w", encoding="utf-8") as w:
    for i, line in enumerate(f, 1):
        obj = json.loads(line)
        if doc_id is None: doc_id = obj.get("doc_id","")
        page = int(obj["page"])
        block_id = obj["block_id"]
        text = obj.get("text","")
        flags = obj.get("flags",[])
        math_latex = []
        if obj.get("type") == "formula" and isinstance(obj.get("latex_consensus",""), str) and obj["latex_consensus"].strip():
            math_latex = [obj["latex_consensus"]]
        chunk = {
            "doc_id": doc_id,
            "chunk_id": f"c{page:03d}_{block_id}",
            "page_start": page,
            "page_end": page,
            "block_ids": [block_id],
            "text": text,
            "math_latex": math_latex,
            "flags": flags,
            "meta": {
                "section_hint": guess_section_hint(text),
                "role": ""
            }
        }
        w.write(json.dumps(chunk, ensure_ascii=False) + "\n")
print(f"WROTE {out}")
