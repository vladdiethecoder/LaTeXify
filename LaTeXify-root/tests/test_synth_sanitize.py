# tests/test_synth_sanitize.py
from __future__ import annotations
import json
from latexify.pipeline.synth_latex import _sanitize_inline, build_snippet

def test_escape_backslash_and_braces():
    s = r"I'm \myage{day}{month}{year} years old."
    out = _sanitize_inline(s)
    assert r"\textbackslash{}myage\{day\}\{month\}\{year\}" in out

def test_escape_underscore_and_percent_and_dollar():
    s = "file_name%price$"
    out = _sanitize_inline(s)
    assert r"file\_name\%price\$" in out

def test_build_snippet_minimal():
    bundle = {
        "task_id": "t00",
        "question": r"How to print \verb-like text?",
        "rubric": ["Include example with $ and _ and \\."],
        "user_answer": {"chunks": [{"text": r"Show \alist(1,2) literally."}]},
    }
    tex = build_snippet(bundle)
    # raw backslash is printable
    assert r"\textbackslash{}alist" in tex
    # \$ and \_ should appear from the rubric bullet
    assert r"\$" in tex and r"\_" in tex
