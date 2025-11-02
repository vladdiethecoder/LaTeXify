# -*- coding: utf-8 -*-
from __future__ import annotations
import io, json, os, re, tempfile
from types import SimpleNamespace
import latexify.pipeline.aggregator as ag

def _mk(tmp: str, name: str, text: str) -> str:
    p = os.path.join(tmp, name)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)
    return p

def test_auto_fallback_to_scrartcl(monkeypatch, tmp_path):
    tmp = str(tmp_path)
    # minimal plan with LiX class
    plan = {
        "doc_class": "lix_article",
        "frontmatter": {"title": "X"},
        "tasks": [{"id": "t01"}, {"id": "t02"}],
    }
    _mk(tmp, "plan.json", json.dumps(plan))

    # two small sanitized snippets
    _mk(tmp, "snippets/t01.tex", "\\section*{Task t01}\\label{sec:t01}\nOK\n")
    _mk(tmp, "snippets/t02.tex", "\\section*{Task t02}\\label{sec:t02}\nOK\n")

    # patch latexmk calls: first fail with class-missing, then succeed
    calls = {"n": 0}
    def fake_run(cmd, cwd=None, capture_output=True, text=True):
        calls["n"] += 1
        if calls["n"] == 1:
            return SimpleNamespace(returncode=1, stdout="", stderr="! LaTeX Error: File `lix_article.cls' not found.")
        return SimpleNamespace(returncode=0, stdout="OK", stderr="")
    monkeypatch.setattr(ag.subprocess, "run", fake_run)

    out_dir = os.path.join(tmp, "build")
    res = ag.aggregate(os.path.join(tmp, "plan.json"), os.path.join(tmp, "snippets"), out_dir, compile_pdf=True)
    assert res["status"] in {"ok_fallback", "compiled_ok"}
    # ensure the class line was rewritten to scrartcl
    main_tex = os.path.join(out_dir, "main.tex")
    main_s = open(main_tex, "r", encoding="utf-8").read()
    assert re.search(r"\\documentclass\[11pt\]\{scrartcl\}", main_s)
