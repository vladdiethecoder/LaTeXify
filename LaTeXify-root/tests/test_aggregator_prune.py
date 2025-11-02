# tests/test_aggregator_prune.py
from __future__ import annotations
import json, os, re
from types import SimpleNamespace
import latexify.pipeline.aggregator as ag

def _mk(tmp: str, path: str, text: str):
    p = os.path.join(tmp, path); os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f: f.write(text)
    return p

def test_prune_missing_package(monkeypatch, tmp_path):
    tmp = str(tmp_path)
    plan = {"doc_class": "scrartcl", "tasks": [{"id": "t01"}]}
    _mk(tmp, "plan.json", json.dumps(plan))
    _mk(tmp, "snippets/t01.tex", "\\section*{Task t01}\\label{sec:t01}\\begin{itemize}\\item OK\\end{itemize}\n")

    # First run: missing microtype.sty, second: OK after prune
    calls = {"n": 0}
    def fake_run(cmd, cwd=None, capture_output=True, text=True):
        calls["n"] += 1
        if calls["n"] == 1:
            return SimpleNamespace(returncode=1, stdout="", stderr="! LaTeX Error: File `microtype.sty' not found.")
        return SimpleNamespace(returncode=0, stdout="OK", stderr="")
    monkeypatch.setattr(ag.subprocess, "run", fake_run)

    out_dir = os.path.join(tmp, "build")
    res = ag.aggregate(os.path.join(tmp, "plan.json"), os.path.join(tmp, "snippets"), out_dir, compile_pdf=True)
    assert res["status"] in {"compiled_ok"}
    s = open(os.path.join(out_dir, "main.tex"), "r", encoding="utf-8").read()
    assert r"\usepackage{microtype}" not in s
