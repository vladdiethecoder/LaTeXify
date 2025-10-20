# scripts/build_chunks.py
from __future__ import annotations
import argparse
from pathlib import Path
from dev.chunking.page_aware_chunker import build_chunks_for_run

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--pdf", type=str, required=True)
    args = ap.parse_args()

    out = build_chunks_for_run(Path(args.run_dir), Path(args.pdf))
    print("Wrote chunks:", out)

if __name__ == "__main__":
    main()
