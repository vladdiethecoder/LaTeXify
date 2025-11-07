from __future__ import annotations

import json
from pathlib import Path

from latexify.pipeline.judge_model import load_council_chunks, run_judge


def _write_backend_chunk(path: Path, backend: str, chunk_id: str, text: str, confidence: float) -> None:
    backend_dir = path / "outputs" / backend
    backend_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "backend": backend,
        "chunk_id": chunk_id,
        "page_index": 0,
        "text": text,
        "confidence": confidence,
        "metadata": {"block_type": "text"},
    }
    (backend_dir / f"{chunk_id}.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_consensus_chunk(path: Path, chunk_id: str, text: str, content_type: str) -> None:
    consensus_dir = path / "consensus"
    consensus_dir.mkdir(parents=True, exist_ok=True)
    (consensus_dir / f"{chunk_id}.json").write_text(
        json.dumps({"chunk_id": chunk_id, "text": text, "content_type": content_type}),
        encoding="utf-8",
    )


def test_load_council_chunks(tmp_path):
    run_dir = tmp_path / "run"
    _write_backend_chunk(run_dir, "internvl", "page0001-chunk001", "Alpha beta", 0.8)
    _write_backend_chunk(run_dir, "florence2", "page0001-chunk001", "Alpha beta", 0.6)
    _write_consensus_chunk(run_dir, "page0001-chunk001", "Alpha beta", "text")
    chunks = load_council_chunks(run_dir, run_dir / "consensus")
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.chunk_id == "page0001-chunk001"
    assert len(chunk.records) == 2
    assert chunk.consensus["content_type"] == "text"


def test_run_judge_emits_blocks(tmp_path):
    run_dir = tmp_path / "run"
    _write_backend_chunk(run_dir, "internvl", "page0001-chunk001", "Reading text", 0.9)
    _write_backend_chunk(run_dir, "mineru", "page0001-chunk001", "col | val", 0.4)
    _write_consensus_chunk(run_dir, "page0001-chunk001", "value | count\n1 | 2", "table")
    out = run_judge(run_dir)
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["block_id"] == "page0001-chunk001"
    assert "judge" in record
    assert record["judge"]["golden_snippet"]["text"]
    golden_file = run_dir / "golden_snippets" / "page0001-chunk001.tex"
    assert golden_file.exists()
    assert "\\begin{table" in golden_file.read_text(encoding="utf-8")
