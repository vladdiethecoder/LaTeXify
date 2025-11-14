#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT:$PYTHONPATH"
python3 -m venv "$ROOT/.venv"
source "$ROOT/.venv/bin/activate"
pip install --upgrade pip
pip install -e "$ROOT"
echo "Environment ready. Activate via 'source .venv/bin/activate'."
