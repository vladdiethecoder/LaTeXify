#!/usr/bin/env python3
"""
Wrapper CLI for the Step‑1 Ingestion Council.

Delegates to ``latexify.ingestion.ingest_pdf`` so that existing shell
scripts can keep invoking ``scripts/ingest_pdf.py`` while the real logic
lives inside the package (and is exposed via ``latexify-ingest-pdf``).
"""

from latexify.ingestion.ingest_pdf import main


if __name__ == "__main__":
    raise SystemExit(main())
