# tests/test_synth_latex.py
from __future__ import annotations
from pathlib import Path
import json

from scripts.synth_latex import synthesize_snippet

def test_synthesize_minimal(tmp_path: Path):
    # Minimal bundle with rubric + user chunks
    b = {
        "task_id": "t01",
        "question": "Align equations and cite units using siunitx. Also include a small table.",
        "rubric": [{"id":"r1","page":1,"text":"Use amsmath's align with clear labels."}],
        "assignment_rules": [{"id":"a1","text":"Follow booktabs style for tables."}],
        "user_answer": {"chunks": [{"id":"u1","text":"We should show a line equation and a divergence law."}], "flags":[]},
    }
    bundle_path = tmp_path / "t01.json"
    bundle_path.write_text(json.dumps(b), encoding="utf-8")
    out_dir = tmp_path / "snippets"

    out = synthesize_snippet(bundle_path, out_dir, kb_dir=None)
    txt = out.read_text(encoding="utf-8")

    assert out.exists()
    assert r"\section*{Task t01:" in txt
    assert r"\label{sec:t01}" in txt
    # mechanics triggered
    assert r"\begin{align}" in txt
    assert r"\SI{9.81}" in txt
    assert r"\begin{table}" in txt
    # no preamble here
    assert r"\documentclass" not in txt
