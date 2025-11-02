"""Convenience entry point for running latexify pipelines without installation."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the repository root is importable when invoked directly.
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from latexify.pipeline.phase2_run_task import main as phase2_main


if __name__ == "__main__":
    phase2_main()
