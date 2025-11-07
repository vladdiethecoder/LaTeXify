#!/usr/bin/env bash
set -euo pipefail

# --- config ---------------------------------------------------------------
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || echo "$PWD")"
ROOT="$REPO_ROOT/LaTeXify-root"
LOG_DIR="$ROOT/out"
SMOKE_LOG="$LOG_DIR/smoke.log"
PY="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$ROOT/.venv}"
EXTRA_PIP_FLAGS="${EXTRA_PIP_FLAGS:-}"

mkdir -p "$LOG_DIR"

log() { printf "[smoke] %s\n" "$*" | tee -a "$SMOKE_LOG"; }

# --- step 0: basic sanity -------------------------------------------------
log "repo_root=$REPO_ROOT"
log "target_root=$ROOT"
if [[ ! -d "$ROOT" ]]; then
  log "ERROR: $ROOT not found. Are you at repo root?"
  exit 2
fi

# --- step 1: python & venv ------------------------------------------------
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "Python not found"; exit 3
fi
log "python=$("$PY" -V 2>&1)"

if [[ ! -d "$VENV_DIR" ]]; then
  log "creating venv at $VENV_DIR"
  "$PY" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
python -m pip install -U pip wheel setuptools $EXTRA_PIP_FLAGS

# --- step 2: install project ---------------------------------------------
cd "$ROOT"
if [[ -f pyproject.toml ]]; then
  # Parse pyproject.toml using tomllib (Py3.11+) to decide path
  if python - <<'PY'
import sys
try:
    import tomllib  # Py3.11+
    tomllib.load(open('pyproject.toml','rb'))
    sys.exit(0)
except Exception:
    sys.exit(1)
PY
  then
    log "pyproject.toml detected → attempting editable install"
    pip install -e ".[dev]" || pip install -e .
  else
    log "pyproject.toml present but not parsable; attempting generic install"
    pip install -e .
  fi
elif compgen -G "requirements*.txt" >/dev/null; then
  log "requirements*.txt detected → pip install -r"
  for f in requirements*.txt; do pip install -r "$f"; done
else
  log "No pyproject/requirements detected — proceeding without project install"
fi

# --- step 3: optional helpers --------------------------------------------
if command -v just >/dev/null 2>&1 && [[ -f "$REPO_ROOT/Justfile" || -f "$ROOT/Justfile" ]]; then
  log "Justfile detected; listing recipes"
  (cd "$REPO_ROOT" && just -l) | tee -a "$SMOKE_LOG" || true
fi
if [[ -f "$REPO_ROOT/Makefile" || -f "$ROOT/Makefile" ]]; then
  log "Makefile detected (not invoked by smoke)"
fi

# --- step 4: module/CLI probes -------------------------------------------
probe() {
  local mod="$1"
  log "probing python -m $mod -h"
  python - <<PY
import runpy, sys
mod = "${mod}"
try:
    try:
        runpy.run_module(mod, run_name="__main__")
    except SystemExit as e:
        print(f"__main__ exited with: {e.code}")
except Exception as exc:
    print(f"IMPORT/MAIN FAILURE for {mod}: {exc}")
    sys.exit(1)
PY
}

log "import sanity: import latexify"
python - <<'PY' || { log "import failed"; exit 4; }
import importlib
m = importlib.import_module("latexify")
print("OK: imported", m.__name__)
PY

for m in \
  "latexify.pipeline.phase2_run_task" \
  "latexify.pipeline.planner_scaffold" \
  "latexify.pipeline.retrieval_bundle" \
  "latexify.pipeline.specialist_router" \
  "latexify.pipeline.critic_agent"
do
  probe "$m" || log "WARN: probe failed for $m"
done

# --- step 5: run existing tests if present -------------------------------
if ls tests/test_*.py >/dev/null 2>&1; then
  log "pytest discovered; running unit tests quietly"
  pip install -U pytest pytest-cov >/dev/null 2>&1 || true
  pytest -q || log "WARN: pytest failures (captured in $SMOKE_LOG)"
else
  log "No tests/* found"
fi

# --- step 6: LaTeX toolchain sanity (tectonic preferred) -----------------
TEX_DIR="$ROOT/out/tex-smoke"
mkdir -p "$TEX_DIR"
TEX_MAIN="$TEX_DIR/smoke.tex"
cat > "$TEX_MAIN" <<'TEX'
\documentclass{article}
\begin{document}
LaTeX toolchain smoke test: \today
\end{document}
TEX

build_with_tectonic() {
  if command -v tectonic >/dev/null 2>&1; then
    log "tectonic found; compiling tex smoke"
    (cd "$TEX_DIR" && tectonic --keep-logs --keep-intermediates smoke.tex) || return 1
    return 0
  fi
  return 1
}

if ! build_with_tectonic; then
  log "tectonic not available; skipping local PDF build (CI will handle via action)"
fi

log "DONE. Artifacts under $ROOT/out"
