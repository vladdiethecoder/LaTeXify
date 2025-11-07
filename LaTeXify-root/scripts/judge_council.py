#!/usr/bin/env python3
"""
Thin wrapper that forwards to ``latexify.pipeline.judge_model`` so existing shell
flows can run ``python scripts/judge_council.py`` without depending on the
installed console entry.
"""

from latexify.pipeline.judge_model import main


if __name__ == "__main__":
    raise SystemExit(main())
