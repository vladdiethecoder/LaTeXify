# scripts/consensus.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Any

def choose_consensus_text(model_texts: Dict[str, str]) -> str:
    """
    Extremely simple stand-in consensus:
      1) prefer the text with the median character length (guards against super short/very long).
      2) if tie, pick the lexicographically first model name’s text.
    This is intentionally basic and deterministic.
    """
    items = [(m, (t or "").strip()) for m, t in model_texts.items()]
    items = [(m, t) for m, t in items if t]
    if not items:
        return ""
    items.sort(key=lambda kv: len(kv[1]))
    mid = (len(items) - 1) // 2
    return items[mid][1]


def save_consensus_for_run(run_dir: Path, per_page: List[Dict[str, Any]]) -> List[Path]:
    """
    Writes consensus .md per page to dev/runs/<stamp>/consensus/page-XXXX.md
    Returns list of paths written.
    """
    out_dir = run_dir / "consensus"
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for page_rec in per_page:
        page_name = page_rec["page"]
        texts: Dict[str, str] = {}
        for r in page_rec.get("results", []):
            md_path = r.get("out_md")
            if md_path:
                try:
                    t = Path(md_path).read_text(encoding="utf-8")
                except Exception:
                    t = ""
                if t:
                    texts[r.get("model", "unknown")] = t
        cons = choose_consensus_text(texts)
        out_path = out_dir / page_name.replace(".png", ".md")
        out_path.write_text(cons, encoding="utf-8")
        written.append(out_path)
    return written
