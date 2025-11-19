import json
from pathlib import Path

from release.core import common
from release.pipeline import reward_suite


def test_reward_evaluation(tmp_path):
    chunks = [
        common.Chunk(chunk_id="c1", page=1, text="Simple harmonic motion describes oscillations.", images=[], metadata={})
    ]
    chunks_path = tmp_path / "chunks.json"
    common.save_chunks(chunks, chunks_path)
    tex_path = tmp_path / "main.tex"
    tex_path.write_text(
        "\\documentclass{article}\n\\usepackage{booktabs}\n\\begin{document}\nSimple harmonic motion describes oscillations.\n\\end{document}",
        encoding="utf-8",
    )
    validation_path = tmp_path / "validation.json"
    validation_path.write_text(json.dumps({"success": True, "errors": []}), encoding="utf-8")
    reward_path = tmp_path / "rewards.json"
    reward_suite.evaluate_rewards(chunks_path, tex_path, validation_path, reward_path)
    payload = json.loads(reward_path.read_text(encoding="utf-8"))
    assert "reward" in payload
    assert payload["components"]["syntax"] == 1.0


def test_reward_modes_and_trace(tmp_path, monkeypatch):
    chunks = [common.Chunk(chunk_id="c2", page=1, text="biology content", images=[], metadata={})]
    chunks_path = tmp_path / "chunks.json"
    common.save_chunks(chunks, chunks_path)
    tex_path = tmp_path / "main.tex"
    tex_path.write_text("\\documentclass{article}\\begin{document}biology\\end{document}", encoding="utf-8")
    validation_path = tmp_path / "validation.json"
    validation_path.write_text(json.dumps({"success": False, "errors": ["missing $"]}), encoding="utf-8")
    reward_path = tmp_path / "rewards.json"
    trace_path = tmp_path / "reward_trace.jsonl"

    def fake_mm_score(_tex_path, pages=None, dpi=0):  # noqa: ARG001 - signature must match adapter
        return 0.42

    monkeypatch.setattr(reward_suite, "aesthetic_mm_score", fake_mm_score)
    reward_suite.evaluate_rewards(
        chunks_path,
        tex_path,
        validation_path,
        reward_path,
        mode="mm",
        trace_path=trace_path,
    )
    payload = json.loads(reward_path.read_text(encoding="utf-8"))
    assert payload["components"]["aesthetic"] == 0.42
    trace_lines = trace_path.read_text(encoding="utf-8").strip().splitlines()
    assert trace_lines and json.loads(trace_lines[-1])["mode"] == "mm"
