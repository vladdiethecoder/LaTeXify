# scripts/build_chunks.py
from __future__ import annotations

import argparse
from pathlib import Path

from dev.chunking.page_aware_chunker import build_chunks_for_run

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--pdf", required=True)
    p.add_argument("--max_chars", type=int, default=1200)
    p.add_argument("--overlap",   type=int, default=150)   # ← fix was here
    p.add_argument("--min_chars", type=int, default=200)
    args = p.parse_args()

    out = build_chunks_for_run(
        Path(args.run_dir),
        Path(args.pdf),
        max_chars=args.max_chars,
        overlap=args.overlap,
        min_par_len=args.min_par_len,
        write_path=(Path(args.out) if args.out else None),
    )
    print(f"Wrote {out}")

if __name__ == "__main__":
    main()
