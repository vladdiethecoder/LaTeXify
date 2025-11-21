from pathlib import Path

from latexify.core import common
from latexify.pipeline import cross_validation as cv


class FakeAdapter:
    def generate(self, prompt, max_tokens=0, temperature=0.0):  # noqa: ARG002
        return "YES"


def test_cross_validation_report(monkeypatch, tmp_path):
    chunks = [
        common.Chunk(chunk_id="c1", page=1, text="This is figure one", images=["fig1.png"], metadata={"region_type": "figure", "layout_confidence": 0.8}),
        common.Chunk(chunk_id="c2", page=1, text="Equation content", metadata={"region_type": "equation", "layout_confidence": 0.9}),
    ]
    tex_path = tmp_path / "doc.tex"
    tex_path.write_text("\\documentclass{article}\\begin{document}\\section{Intro}\\includegraphics{fig1.png}\\begin{equation}x=1\\end{equation}\\end{document}", encoding="utf-8")
    pdf_path = tex_path.with_suffix(".pdf")
    pdf_path.write_text("pdf", encoding="utf-8")
    monkeypatch.setattr(cv, "get_kimi_adapter", lambda: FakeAdapter())
    report = cv.run_cross_validation(chunks, tex_path.read_text(), {}, pdf_path=pdf_path)
    payload = report.to_dict()
    assert 0 <= payload["overall_score"] <= 1
    assert payload["structural"]["composite"] > 0.5
    assert payload["semantic"]["samples"] >= 0
