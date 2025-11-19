from pathlib import Path

from release.pipeline.robust_compilation import run_robust_compilation, RobustCompiler


def _dummy_engine(self, tex_file: Path, *, fallback: bool = False):  # noqa: ARG001
    pdf = tex_file.with_suffix(".pdf")
    pdf.write_text("pdf", encoding="utf-8")
    log = tex_file.with_suffix(".log")
    log.write_text("", encoding="utf-8")
    return True, "dummy"


def test_run_robust_compilation_uses_cache(tmp_path, monkeypatch):
    tex_path = tmp_path / "doc.tex"
    tex_path.write_text(
        "\\documentclass{article}\n\\begin{document}\n\\section{Intro}Hello\\section{Body}World\\end{document}",
        encoding="utf-8",
    )
    monkeypatch.setattr(RobustCompiler, "_run_engine", _dummy_engine)
    result = run_robust_compilation(tex_path)
    assert result.success
    assert result.pdf_path and result.pdf_path.exists()
    # Modify only second section to trigger selective recompilation
    updated = tex_path.read_text(encoding="utf-8").replace("World", "Updated World")
    tex_path.write_text(updated, encoding="utf-8")
    result2 = run_robust_compilation(tex_path)
    assert result2.success
    assert result2.pdf_path and result2.pdf_path.exists()
