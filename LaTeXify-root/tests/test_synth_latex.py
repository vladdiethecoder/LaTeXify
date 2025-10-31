# tests/test_synth_latex.py
from __future__ import annotations
import json
from pathlib import Path

from scripts.synth_latex import synthesize_snippet

def test_synth_section_snippet(tmp_path: Path):
    bundle = {
        "task_id": "T03",
        "question": "T03: Introduction",
        "rubric": [{"id":"r1","text":"Explain background and goals."}],
        "assignment_rules": [],
        "assessment": [],
        "user_answer": {
            "chunks": [{"id":"u1","text":"State the problem clearly and motivate its importance."}],
            "flags": {"ocr_uncertain": False}
        }
    }
    s = synthesize_snippet(bundle)
    assert "\\section{Introduction}" in s
    assert "\\label{sec:T03-introduction}" in s
    assert "State the problem clearly" in s

def test_synth_abstract_snippet():
    bundle = {
        "task_id": "T02",
        "question": "T02: Abstract",
        "rubric": [],
        "assignment_rules": [],
        "assessment": [],
        "user_answer": {"chunks": [], "flags": {"ocr_uncertain": True}},
    }
    s = synthesize_snippet(bundle)
    assert "\\begin{abstract}" in s and "\\end{abstract}" in s
    assert "\\todo{Verify OCR" in s
