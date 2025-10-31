#!/usr/bin/env python
import sys, json, ast, argparse

p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
args = p.parse_args()

buf = []
depth = 0
kept = 0

def try_emit(s, w):
    global kept
    s = s.strip()
    if not s:
        return
    # Try strict JSON first
    try:
        obj = json.loads(s)
    except Exception:
        # Fall back to Python dict style -> JSON
        try:
            obj = ast.literal_eval(s)
        except Exception:
            return
    # Minimal schema guard
    need = ("doc_id","page","block_id","type","text","flags")
    if not all(k in obj for k in need):
        return
    w.write(json.dumps(obj, ensure_ascii=False) + "\n")
    kept += 1

with open(args.out, "w", encoding="utf-8") as w:
    for line in sys.stdin:
        # Scan char-by-char to capture multi-line {...} blocks
        for ch in line:
            if ch == '{':
                depth += 1
            if depth > 0:
                buf.append(ch)
            if ch == '}':
                depth -= 1
                if depth == 0:
                    try_emit(''.join(buf), w)
                    buf = []
    # Flush any trailing object
    if depth == 0 and buf:
        try_emit(''.join(buf), w)

print(f"JSONL kept={kept} -> {args.out}", file=sys.stderr)
