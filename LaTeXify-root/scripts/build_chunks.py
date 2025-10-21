# scripts/build_chunks.py
from __future__ import annotations
import argparse
from pathlib import Path

from dev.chunking.page_aware_chunker import build_chunks_for_run

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True, help="Path to run dir (with /pages and /outputs).")
    p.add_argument("--pdf", required=True, help="Source PDF path.")
    p.add_argument("--max_chars", type=int, default=800)
    p.add_argument("--overlap", type=int, default=120)
    p.add_argument("--min_par_len", type=int, default=60)
    args = p.parse_args()

    build_chunks_for_run(
        run_dir=Path(args.run_dir),
        pdf_path=Path(args.pdf),
        max_chars=args.max_chars,
        overlap=args.overlap,
        min_par_len=args.min_par_len,
    )

if __name__ == "__main__":
    main()
