from __future__ import annotations

import sys
from pathlib import Path

import pytest

from latexify.ingestion.internvl_hf_adapter import InternVLHFAdapter

MOCK_RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "mock_internvl_hf.py"


def _build_adapter(tmp_path, extra_args=None, **overrides):
    params = dict(
        runner_script=MOCK_RUNNER,
        model_dir=tmp_path,
        python_executable=sys.executable,
        max_new_tokens=32,
        extra_args=extra_args or [],
        timeout_seconds=5,
    )
    params.update(overrides)
    return InternVLHFAdapter(**params)


def test_hf_adapter_parses_json_payload(tmp_path):
    adapter = _build_adapter(tmp_path, ["--mode", "json"])
    text, metadata = adapter.generate("describe JSON", None)
    assert text == "JSON:describe JSON"
    assert metadata["format"] == "json"
    assert metadata["raw_json"]["prompt"] == "describe JSON"


def test_hf_adapter_parses_text_block(tmp_path):
    adapter = _build_adapter(tmp_path)
    text, metadata = adapter.generate("plain text", None)
    assert text.startswith("TEXT:plain text")
    assert metadata["format"] == "text"


def test_hf_adapter_raises_on_nonzero_exit(tmp_path):
    adapter = _build_adapter(tmp_path, ["--fail-code", "3"])
    with pytest.raises(RuntimeError) as excinfo:
        adapter.generate("boom", None)
    assert "exit 3" in str(excinfo.value)


def test_hf_adapter_retries_and_succeeds(tmp_path):
    state_file = tmp_path / "state.txt"
    adapter = _build_adapter(
        tmp_path,
        [
            "--mode",
            "json",
            "--fail-code",
            "9",
            "--fail-once",
            "--state-file",
            str(state_file),
        ],
        retries=2,
    )
    text, metadata = adapter.generate("retry me", None)
    assert text == "JSON:retry me"
    # Second attempt should succeed
    assert metadata["attempt"] == 2
