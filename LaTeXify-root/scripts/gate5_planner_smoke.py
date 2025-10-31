#!/usr/bin/env python
import json, os, re, faiss, argparse, textwrap

p = argparse.ArgumentParser()
p.add_argument("--chunks", default="runs/sample/chunks.jsonl")
p.add_argument("--index",  default="runs/sample/index-bgem3/faiss.index")
p.add_argument("--k", type=int, default=5)
p.add_argument("--out", default="runs/sample/planner_smoke.tex")
args = p.parse_args()

def load_chunks(fp):
    rows = []
    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            rows.append(json.loads(line))
    by_id = {r["chunk_id"]: r for r in rows}
    texts = [r["text"] for r in rows]
    return rows, by_id, texts

rows, by_id, texts = load_chunks(args.chunks)
idx = faiss.read_index(args.index)

def synth_from_hits(hits):
    body = []
    flagged = False
    for cid in hits:
        r = by_id[cid]
        t = r.get("text","").strip()
        if any(r.get("flags", [])):
            flagged = True
        # keep it tiny, LiX friendly
        t = re.sub(r"\n{3,}", "\n\n", t)
        body.append(t)
    note = "\\todo{Verify OCR disagreements}" if flagged else ""
    return textwrap.dedent(f"""
    % --- synthesized snippet (Gate 5 smoke) ---
    \\section*{{Planner Retrieval Snippet}}
    {note}
    \\begin{{flushleft}}
    {"\\par\\medskip\n".join(body[:3])}
    \\end{{flushleft}}
    """).strip()

def topk(query, k):
    # trivial embed proxy: reuse FAISS ids as if texts already embedded (we only smoke-test retrieval wiring here)
    # In full pipeline, you’ll embed the query with BGE-M3 and search. For smoke, we pick seed docs by heuristics.
    qs = query.lower()
    order = []
    for r in rows:
        score = (qs in r["text"].lower()) + (qs.split(" ")[0] in r["text"].lower())
        order.append((score, r["chunk_id"]))
    order.sort(reverse=True)
    return [cid for _, cid in order[:k]]

queries = ["Abstract", "Question 1", "Transformations"]
sections = []
for q in queries:
    hits = topk(q, args.k)
    sections.append(synth_from_hits(hits))

os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "w", encoding="utf-8") as w:
    w.write("\n\n".join(sections) + "\n")
print("WROTE", args.out)
