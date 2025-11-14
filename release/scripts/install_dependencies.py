#!/usr/bin/env python3
"""Install LaTeXify Python dependencies with a single command."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REQUIREMENTS = REPO_ROOT / "release" / "requirements.txt"
DEFAULT_EXTRAS = [
    "PyMuPDF>=1.24.0",
]


def run(cmd: List[str]) -> None:
    print("[install-deps]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def install_requirements(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    run([sys.executable, "-m", "pip", "install", "-r", str(path)])


def install_extras(packages: Iterable[str]) -> None:
    pkgs = [pkg for pkg in packages if pkg]
    if not pkgs:
        return
    run([sys.executable, "-m", "pip", "install", *pkgs])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--requirements",
        default=str(DEFAULT_REQUIREMENTS),
        help="Path to the requirements.txt file (default: release/requirements.txt)",
    )
    parser.add_argument(
        "--no-extras",
        action="store_true",
        help="Skip installing optional extras (PyMuPDF, etc.).",
    )
    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        help="Additional packages to install after the main requirements (can be repeated).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    install_requirements(Path(args.requirements))
    extras: List[str] = [] if args.no_extras else list(DEFAULT_EXTRAS)
    extras.extend(args.extra)
    install_extras(extras)
    print("[install-deps] All dependencies installed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
