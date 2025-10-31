#!/usr/bin/env python3
"""
Block-level LaTeX consensus & flagging.

Inputs
- --run_dir: the run folder (defaults to newest under dev/runs/*)
- --layout:  linked_pages.jsonl (defaults to <run_dir>/layout/linked_pages.jsonl)
- --backends: comma-list of backend names to consider (defaults to discovered outputs)
- Assumes per-backend page outputs exist at <run_dir>/outputs/<backend>/page-XXXX.md

Output
- <run_dir>/blocks_refined.jsonl  (one JSON object per block with fields:
  block_id, bbox, block_type, ocr_outputs{backend->string}, latex_consensus,
  agreement_score, latex_agreement_score, flagged, flag_reasons)
"""

from __future__ import annotations
import argparse, json, os, re, sys
from pathlib import Path
from itertools import combinations
from typing import Dict, List, Tuple, Optional

# ----------------------- helpers: filesystem -----------------------

def newest_run_dir(root: Path) -> Path:
    runs = sorted((root / "dev" / "runs").glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        print("No runs found under dev/runs/", file=sys.stderr)
        sys.exit(2)
    return runs[0]

def discover_backends(run_dir: Path) -> List[str]:
    outs = run_dir / "outputs"
    if not outs.exists():
        return []
    return sorted([p.name for p in outs.iterdir() if p.is_dir()])

def load_page_texts(run_dir: Path, backend: str) -> Dict[str, str]:
    """Return {page_name:'page-0001.png' -> page_text} for a backend from page-XXXX.md files."""
    out = {}
    bdir = run_dir / "outputs" / backend
    if not bdir.exists():
        return out
    for md in sorted(bdir.glob("page-*.md")):
        page = md.stem + ".png"  # 'page-0001.md' -> 'page-0001.png'
        try:
            out[page] = md.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            out[page] = ""
    return out

# ----------------------- helpers: text metrics -----------------------

def levenshtein(a: str, b: str) -> int:
    # Classic DP; O(len(a)*len(b)) but fine for short strings
    if a == b: return 0
    if not a: return len(b)
    if not b: return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins = prev[j] + 1
            delete = cur[j-1] + 1
            sub = prev[j-1] + (ca != cb)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]

def norm_space(s: str) -> str:
    # minimal, LaTeX-safe whitespace normalization
    return re.sub(r"\s+", " ", s).strip()

def cer_pair(a: str, b: str) -> float:
    a_n = norm_space(a)
    b_n = norm_space(b)
    if not a_n and not b_n:
        return 0.0
    dist = levenshtein(a_n, b_n)
    denom = max(1, len(a_n))
    return dist / denom

def avg_pairwise(fn, arr: List[str]) -> Optional[float]:
    pairs = [fn(x, y) for x, y in combinations(arr, 2) if x is not None and y is not None]
    if not pairs:
        return None
    return sum(pairs) / len(pairs)

# ----------------------- helpers: LaTeX extraction -----------------------

LATEX_PATTERNS = [
    # order matters: $$ .. $$ first to avoid early \( .. \) matches inside
    (re.compile(r"\$\$(.+?)\$\$", re.DOTALL), "$$"),
    (re.compile(r"\\\[(.+?)\\\]", re.DOTALL), r"\[ \]"),
    (re.compile(r"\\\((.+?)\\\)", re.DOTALL), r"\( \)"),
]

def extract_latex_segments(text: str) -> List[str]:
    if not text:
        return []
    segs = []
    for rx, _kind in LATEX_PATTERNS:
        for m in rx.finditer(text):
            segs.append(m.group(0))  # keep delimiters in the raw string
    return segs

def strip_delims(s: str) -> str:
    """Remove the outermost $$..$$ or \[..\] or \(..\) once; keep inside intact."""
    if s.startswith("$$") and s.endswith("$"):
        # naive: '$$...$$'
        return s[2:-2]
    if s.startswith(r"\[") and s.endswith(r"\]"):
        return s[2:-2]
    if s.startswith(r"\(") and s.endswith(r"\)"):
        return s[2:-2]
    return s

def normalize_latex_for_compare(s: str) -> str:
    core = strip_delims(s)
    # minimal normalization: collapse whitespace; remove spaces before {} , ^ _
    core = re.sub(r"\s+", " ", core)
    core = re.sub(r"\s+([{}^_])", r"\1", core)
    core = core.strip()
    return core

def latex_similarity(a: str, b: str) -> float:
    """Return normalized distance (0.0 exact, 1.0 max) on normalized LaTeX bodies."""
    an, bn = normalize_latex_for_compare(a), normalize_latex_for_compare(b)
    if not an and not bn:
        return 0.0
    dist = levenshtein(an, bn)
    denom = max(1, max(len(an), len(bn)))
    return dist / denom

def latex_balanced(s: str) -> bool:
    body = strip_delims(s)
    # naive brace balance check
    return body.count("{") == body.count("}") and body.count("(") == body.count(")")

# ----------------------- consensus policy -----------------------

PRIORITY = ["nanonets-ocr2-3b", "qwen2-vl-ocr-2b", "nanonets-ocr-s"]
DISAGREE_THR = 0.15   # general CER disagreement
LATEX_CLOSE_THR = 0.10
ARTIFACT_PAT = re.compile(r"<\|im_.*?\|>")

def choose_latex_consensus(cands: Dict[str, List[str]]) -> Tuple[Optional[str], Optional[float], List[str]]:
    """cands: backend -> list of raw LaTeX segments (with delimiters)."""
    flag_reasons: List[str] = []
    # flatten into list of (backend, raw)
    flat: List[Tuple[str, str]] = []
    for be, lst in cands.items():
        for s in lst:
            flat.append((be, s))
    if not flat:
        return None, None, ["LaTeX Missing"]
    # measure pairwise distances on normalized cores
    if len(flat) == 1:
        s = flat[0][1]
        flags = []
        if not latex_balanced(s): flags.append("Malformed LaTeX")
        return s, 0.0, flags
    # look for a close pair
    best_pair = (1.0, None, None)  # (dist, idx_i, idx_j)
    for i in range(len(flat)):
        for j in range(i+1, len(flat)):
            d = latex_similarity(flat[i][1], flat[j][1])
            if d < best_pair[0]:
                best_pair = (d, i, j)
    avg_pw = None
    # compute avg pairwise on the top 3 unique strings (fallback)
    uniq = list({normalize_latex_for_compare(s): s for _, s in flat}.values())
    if len(uniq) >= 2:
        avg_pw = avg_pairwise(latex_similarity, uniq)

    if best_pair[0] <= LATEX_CLOSE_THR:
        # choose longer raw among the close pair
        i, j = best_pair[1], best_pair[2]
        assert i is not None and j is not None
        s_i = flat[i][1]; s_j = flat[j][1]
        pick = s_i if len(strip_delims(s_i)) >= len(strip_delims(s_j)) else s_j
        flags = []
        if not latex_balanced(pick): flags.append("Malformed LaTeX")
        return pick, (avg_pw if avg_pw is not None else best_pair[0]), flags

    # otherwise pick by backend priority, falling back to longest
    best = None
    for be in PRIORITY:
        cand = [s for (b, s) in flat if b == be]
        if cand:
            best = max(cand, key=lambda x: len(strip_delims(x)))
            break
    if best is None:
        best = max((s for (_b, s) in flat), key=lambda x: len(strip_delims(x)))
    flags = ["LaTeX Disagreement"]
    if not latex_balanced(best): flags.append("Malformed LaTeX")
    return best, avg_pw, flags

def compute_agreement_score(variants: List[str]) -> Optional[float]:
    keep = [v for v in variants if v]
    if len(keep) < 2: return None
    return avg_pairwise(cer_pair, keep)

# ----------------------- main pipeline -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, default=None, help="Run folder (defaults to newest dev/runs/*)")
    ap.add_argument("--layout", type=str, default=None, help="linked_pages.jsonl (defaults to <run_dir>/layout/linked_pages.jsonl)")
    ap.add_argument("--backends", type=str, default=None, help="Comma list of backends (defaults to discovered outputs)")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]  # scripts/ -> repo root
    run_dir = Path(args.run_dir) if args.run_dir else newest_run_dir(repo_root)
    layout_path = Path(args.layout) if args.layout else (run_dir / "layout" / "linked_pages.jsonl")
    if not layout_path.exists():
        print(f"Layout not found: {layout_path}", file=sys.stderr); sys.exit(2)

    # discover or parse backends
    backs = []
    if args.backends:
        backs = [b.strip() for b in args.backends.split(",") if b.strip()]
    else:
        backs = discover_backends(run_dir)
    if not backs:
        print("No backends discovered under outputs/; nothing to do.", file=sys.stderr)
        sys.exit(2)

    # load page texts
    page_texts: Dict[str, Dict[str, str]] = {}
    for be in backs:
        pt = load_page_texts(run_dir, be)
        for page, txt in pt.items():
            page_texts.setdefault(page, {})[be] = txt

    out_path = run_dir / "blocks_refined.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_blocks = 0; n_flagged = 0
    with layout_path.open("r", encoding="utf-8", errors="ignore") as f_in, \
         out_path.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            if not line.strip(): continue
            rec = json.loads(line)
            # attempt to read page id and blocks array with bboxes and block type
            page = rec.get("page") or rec.get("page_id") or rec.get("image") or rec.get("name")
            blocks = rec.get("blocks") or rec.get("regions") or []
            if not page or not isinstance(blocks, list):
                # passthrough: nothing to refine on this line
                continue

            # page-level artifacts for later flagging
            page_artifacts = []
            for be, txt in page_texts.get(page, {}).items():
                if ARTIFACT_PAT.search(txt or ""):
                    page_artifacts.append(be)

            for bi, b in enumerate(blocks):
                n_blocks += 1
                bbox = b.get("bbox") or b.get("box")
                btype = (b.get("block_type") or b.get("type") or "").strip() or "Text"
                block_id = b.get("block_id") or f"{Path(page).stem}-block-{bi:03d}"

                # collect candidate strings per backend
                ocr_outputs: Dict[str, str] = {}
                for be in backs:
                    page_txt = page_texts.get(page, {}).get(be, "")
                    ocr_outputs[be] = page_txt

                flag_reasons: List[str] = []
                if page_artifacts:
                    flag_reasons.append(f"Model Artifact: {','.join(page_artifacts)}")

                latex_consensus = None
                latex_agree = None
                agreement_score = None

                # If this is a formula/maths block, compute LaTeX consensus
                if btype.lower() in {"formula", "equation", "math", "displaymath"}:
                    per_backend_segments: Dict[str, List[str]] = {
                        be: extract_latex_segments(txt) for be, txt in ocr_outputs.items()
                    }
                    latex_consensus, latex_agree, lflags = choose_latex_consensus(per_backend_segments)
                    if lflags:
                        flag_reasons.extend(lflags)
                else:
                    # basic general agreement on page-level text (coarse)
                    agreement_score = compute_agreement_score(list(ocr_outputs.values()))
                    if (agreement_score is not None) and (agreement_score > DISAGREE_THR):
                        flag_reasons.append("High OCR Disagreement")

                flagged = bool(flag_reasons)

                out_obj = {
                    "block_id": block_id,
                    "bbox": bbox,
                    "block_type": btype,
                    "page": page,
                    "ocr_outputs": ocr_outputs,  # page-level text per backend
                    "latex_consensus": latex_consensus,
                    "agreement_score": agreement_score,
                    "latex_agreement_score": latex_agree,
                    "flagged": flagged,
                    "flag_reasons": flag_reasons,
                }
                f_out.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                if flagged: n_flagged += 1

    print(json.dumps({
        "run_dir": str(run_dir),
        "blocks_refined": str(out_path),
        "blocks_total": n_blocks,
        "blocks_flagged": n_flagged
    }, indent=2))

if __name__ == "__main__":
    main()
