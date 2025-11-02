# tests/test_aggregator_missing_plan.py
from __future__ import annotations
from pathlib import Path
import pytest
from latexify.pipeline.aggregator import run_aggregator

def test_missing_plan_raises(tmp_path: Path):
    out_dir = tmp_path / "build"
    with pytest.raises(SystemExit):
        run_aggregator(
            tmp_path / "does_not_exist.json",
            tmp_path / "snippets",
            out_dir,
            no_compile=True,
            simulate=True,
            assets_dir=None,
        )
