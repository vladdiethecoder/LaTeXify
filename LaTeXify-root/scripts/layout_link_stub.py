#!/usr/bin/env python3
r"""
Minimal layout linker (stub) that builds block lists from existing OCR outputs.
- Inputs: <run_dir>/outputs/<backend>/page-XXXX.md
- Output: <run_dir>/layout/linked_pages.jsonl with blocks labeled as Text/Header/Formula.
This is a heuristic placeholder until a vision-layout model is wired in.
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
from typing import Dict, List, Optional

# -------- LaTeX detection --------
# Detect $$...$$, \[...\], \(...\) (raw strings to avoid escape warnings)
RX_DOLLAR = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
RX_BRACK  = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)
RX_PAREN  = re.compile(r"\\\((.+?)\\\)", re.DOTALL)
LATEX_RXS = [RX_DOLLAR, RX_BRACK, RX_PAREN]

def find_latex_segments(text: str) -> List[str]:
    segs: List[str] = []
    for rx in LATEX_RXS:
        segs.extend([m.group(0) for m in rx.finditer(text or "")])
    return segs

def remove_latex_segments(text: str) -> str:
    t = text
    for rx in LATEX_RXS:
        t = rx.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()

# -------- Filesystem helpers --------
def newest_run_dir(repo_root: Path) -> Path:
    runs = sorted((repo_root / "dev" / "runs").glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        print("No runs in dev/runs/", file=sys.stderr); sys.exit(2)
    return runs[0]

def discover_backends(run_dir: Path) -> List[str]:
    outs = run_dir / "outputs"
    return sorted([p.name for p in outs.iterdir() if p.is_dir()]) if outs.exists() else []

def load_pages(run_dir: Path, backend: str) -> Dict[str, str]:
    d = run_dir / "outputs" / backend
    pages: Dict[str, str] = {}
    if not d.exists(): return pages
    for md in sorted(d.glob("page-*.md")):
        pages[md.stem + ".png"] = md.read_text(encoding="utf-8", errors="ignore")
    return pages

# -------- Heuristics --------
def is_header_line(line: str) -> bool:
    s = line.strip()
    if not s: return False
    if s.startswith("#"): return True
    if len(s) <= 80 and (s.istitle() or s.isupper()): return True
    if s.startswith(("Section", "Chapter", "Appendix")): return True
    return False

def para_blocks_from_text(page_text: str) -> List[Dict]:
    blocks: List[Dict] = []
    # Split on blank lines (2+ newlines)
    paras = re.split(r"\n\s*\n", page_text or "")
    idx = 0
    for para in paras:
        para = para.strip()
        if not para: continue

        # Extract LaTeX segments (each becomes a Formula block)
        latex_segs = find_latex_segments(para)
        for seg in latex_segs:
            blocks.append({
                "block_id": None, "bbox": None, "block_type": "Formula",
                "text": seg
            })
            idx += 1

        # Residual paragraph text (with LaTeX removed)
        residual = remove_latex_segments(para)
        if residual:
            btype = "Header" if any(is_header_line(l) for l in residual.splitlines()) else "Text"
            blocks.append({
                "block_id": None, "bbox": None, "block_type": btype,
                "text": residual
            })
            idx += 1
    return blocks

# -------- Main --------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, default=None, help="Run folder (defaults to newest dev/runs/*)")
    ap.add_argument("--prefer", type=str, default="nanonets-ocr2-3b,qwen2-vl-ocr-2b,nanonets-ocr-s",
                    help="Comma list of backends to prefer for page text")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]  # scripts/ -> repo root
    run_dir = Path(args.run_dir) if args.run_dir else newest_run_dir(repo_root)
    prefer = [p.strip() for p in args.prefer.split(",") if p.strip()]
    backends = discover_backends(run_dir)
    if not backends:
        print(f"No outputs/* backends found in {run_dir}", file=sys.stderr); sys.exit(2)

    # Load page texts per backend
    per_be_pages: Dict[str, Dict[str, str]] = {be: load_pages(run_dir, be) for be in backends}
    # Determine page set (union over backends)
    page_names = sorted(set().union(*[set(d.keys()) for d in per_be_pages.values()]))

    layout_dir = run_dir / "layout"
    layout_dir.mkdir(parents=True, exist_ok=True)
    out_path = layout_dir / "linked_pages.jsonl"

    total_blocks = 0
    with out_path.open("w", encoding="utf-8") as f_out:
        for page in page_names:
            # Choose text from preferred backend order
            chosen_text: Optional[str] = None
            for be in prefer:
                t = per_be_pages.get(be, {}).get(page)
                if t:
                    chosen_text = t; break
            if chosen_text is None:
                # fallback to any backend that has this page
                for be in backends:
                    t = per_be_pages.get(be, {}).get(page)
                    if t:
                        chosen_text = t; break
            chosen_text = chosen_text or ""

            blocks = para_blocks_from_text(chosen_text)
            # finalize block ids and write
            for i, b in enumerate(blocks):
                b["block_id"] = f"{Path(page).stem}-block-{i:03d}"
            rec = {"page": page, "blocks": blocks}
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            total_blocks += len(blocks)

    print(json.dumps({
        "run_dir": str(run_dir),
        "layout": str(out_path),
        "pages": len(page_names),
        "blocks": total_blocks,
        "backends_seen": backends,
        "prefer_order": prefer
    }, indent=2))

if __name__ == "__main__":
    main()
