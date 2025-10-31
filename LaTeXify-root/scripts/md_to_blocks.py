#!/usr/bin/env python
import argparse, json, pathlib, re

p = argparse.ArgumentParser()
p.add_argument("--md_dir", required=True)
p.add_argument("--doc_id", required=True)
p.add_argument("--out", required=True)
args = p.parse_args()

md_dir = pathlib.Path(args.md_dir)
out = pathlib.Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)

# page-0001.md -> page number
rx = re.compile(r"page-(\d+)\.md$", re.I)

with out.open("w", encoding="utf-8") as w:
    for md in sorted(md_dir.glob("page-*.md")):
        m = rx.search(md.name)
        if not m: continue
        page = int(m.group(1))
        text = md.read_text(encoding="utf-8", errors="ignore").strip()
        obj = {
            "doc_id": args.doc_id,
            "page": page,
            "block_id": f"p{page:03d}_b001",
            "type": "text",
            "text": text,
            "flags": []
        }
        w.write(json.dumps(obj, ensure_ascii=False) + "\n")
print(f"WROTE {args.out}")
