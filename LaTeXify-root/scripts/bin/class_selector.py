#!/usr/bin/env python3
import argparse, json, re
from pathlib import Path

def decide(chunks_path: Path) -> dict:
    pages = set()
    n_code = n_thm = 0
    for line in chunks_path.open():
        ch = json.loads(line)
        pages.add(ch.get("page"))
        t = (ch.get("text") or "").lower()
        if re.search(r"\\begin\{(theorem|lemma|proof)\}|\\(theorem|lemma)\b", t): n_thm += 1
        if re.search(r"```|\\begin\{minted\}|\\begin\{lstlisting\}", t): n_code += 1
    bias_textbook = len(pages) >= 20 or n_thm >= 8
    cls = "lix_textbook" if bias_textbook else "lix_article"
    pkgs = ["microtype", "amsmath", "amsthm", "mathtools"]
    code = {"use_minted": bool(n_code)}
    if code["use_minted"]: pkgs += ["xcolor", "fvextra", "minted"]
    return {
        "class": cls,
        "packages": pkgs,
        "code": code,
        "signals": {"pages": len(pages), "n_theoremish": n_thm, "n_codeish": n_code}
    }

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--out", default="synthesis/class_decision.json")
    args = ap.parse_args()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    dec = decide(Path(args.chunks))
    out.write_text(json.dumps(dec, indent=2))
    print(json.dumps(dec, indent=2))
