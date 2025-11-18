#!/usr/bin/env python3
"""One-click helper to build the training route and pull datasets."""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def main() -> None:
    run(["python", "scripts/build_training_data_route.py"])
    run(["python", "scripts/prepare_training_data.py"])
    print("Training data route rebuilt and data preparation completed.")


if __name__ == "__main__":
    main()
