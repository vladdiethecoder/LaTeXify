#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run a single page through an OCR backend and print Markdown.

Usage (module mode, recommended):
  PYTHONPATH=. python -m dev.ocr_backends.ocr_page \
    --backend qwen \
    --src "dev/inputs/Basic Skills Review Unit USER.pdf" \
    --page 7

Usage (direct script, works with sys.path injection below):
  python dev/ocr_backends/ocr_page.py \
    --backend trocr-base \
    --src "dev/runs/user_e2e/pages/page_0001.png"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# --- Make running this file directly work like `-m dev.ocr_backends.ocr_page` ---
# Ensure the repository root (two levels up from this file) is on sys.path
# so that `import dev.ocr_backends.*` resolves.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
# -------------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--backend",
        choices=["qwen", "trocr-base", "trocr-small"],
        required=True,
        help="OCR backend to use",
    )
    ap.add_argument("--src", required=True, help="Path to image or PDF")
    ap.add_argument("--page", type=int, default=1, help="1-based page number for PDFs")
    args = ap.parse_args()

    if args.backend == "qwen":
        from dev.ocr_backends.qwen2vl_ocr2b import Backend  # type: ignore
        backend = Backend()
    elif args.backend == "trocr-base":
        from dev.ocr_backends.nanonets_ocr2 import Backend  # type: ignore
        backend = Backend()
    else:
        from dev.ocr_backends.nanonets_s import Backend  # type: ignore
        backend = Backend()

    res = backend.recognize_page(args.src, page=args.page)
    print(res.text_md)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ocr_page] error: {e}", file=sys.stderr)
        sys.exit(1)
