#!/usr/bin/env python3
"""Compatibility shim for latexify.pipeline.aggregator."""
from latexify.pipeline.aggregator import main

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
