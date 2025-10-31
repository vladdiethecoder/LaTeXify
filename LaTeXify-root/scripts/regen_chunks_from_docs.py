# scripts/regen_chunks_from_docs.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

LOG_PATH = Path("kb/latex/build_latex_kb.log.jsonl")
DEFAULT_DOCS = Path("kb/latex/latex_docs.jsonl")


def log_event(event: str, **details) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    rec = {"event": event, **details}
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_jsonl(p: Path) -> List[Dict]:
    if not p.exists():
        return []
    rows: List[Dict] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                # tolerate a bad row and keep going
                pass
    return rows


def make_chunk_rows(docs: List[Dict]) -> List[Dict]:
    out: List[Dict] = []
    for i, rec in enumerate(docs, start=1):
        title = (rec.get("title") or "").strip()
        q = (rec.get("question") or "").strip()
        a = (rec.get("answer") or "").strip()
        code = rec.get("code_blocks") or []
        code_excerpt = "\n".join(code[:1]).strip()
        parts = [x for x in (title, q, a, code_excerpt) if x]
        if not parts:
            continue
        text = "\n\n".join(parts)
        out.append(
            {
                "id": f"d{i:04d}",
                "text": text,
                "page": None,
                "label": "kb",
                "source_image": rec.get("url"),
            }
        )
    # deterministic stable order
    out.sort(key=lambda r: r["id"])
    return out


def safe_write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # If destination is a symlink, unlink so we write a real file.
    if path.is_symlink():
        try:
            path.unlink()
        except Exception:
            # last resort: move the link aside
            path.rename(path.with_suffix(path.suffix + ".oldlink"))
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Regenerate KB chunks from latex_docs.jsonl")
    ap.add_argument("--docs", type=Path, default=DEFAULT_DOCS, help="Path to latex_docs.jsonl")
    ap.add_argument("--out_dir", type=Path, default=Path("kb/latex"), help="KB output dir")
    args = ap.parse_args(argv)

    docs = read_jsonl(args.docs)
    rows = make_chunk_rows(docs)

    # Write both layouts our tooling expects:
    out_dir = args.out_dir
    chunks_dir = out_dir / "chunks" / "chunks.jsonl"
    chunks_single = out_dir / "chunks.jsonl"
    safe_write_jsonl(chunks_dir, rows)
    safe_write_jsonl(chunks_single, rows)

    log_event("regen_chunks_done", rows_total=len(docs), rows_chunks=len(rows),
              files=[str(chunks_dir), str(chunks_single)])
    print(json.dumps({"rows_total": len(docs), "rows_chunks": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
