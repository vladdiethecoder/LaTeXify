import importlib
import subprocess
import sys
from pathlib import Path

def run_cli_help(module: str) -> int:
    # Try invoking as a module; non-zero still proves importable/argparse wired
    proc = subprocess.run([sys.executable, "-m", module, "-h"], capture_output=True)
    # Accept 0 or 2 (argparse can exit 2 when help hits custom parsing)
    return proc.returncode if proc.returncode in (0, 2) else -1

def test_import_package():
    importlib.import_module("latexify")

def test_cli_phase2_help():
    assert run_cli_help("latexify.pipeline.phase2_run_task") in (0, 2)

def test_cli_planner_help():
    assert run_cli_help("latexify.pipeline.planner_scaffold") in (0, 2)

def test_cli_router_imports():
    importlib.import_module("latexify.pipeline.specialist_router")

def test_retrieval_bundle_imports():
    importlib.import_module("latexify.pipeline.retrieval_bundle")
