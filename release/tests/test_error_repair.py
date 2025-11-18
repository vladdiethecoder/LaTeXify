from pathlib import Path

from release.pipeline.error_repair import LaTeXErrorRepair


def _write(tmp_path: Path, name: str, payload: str) -> Path:
    target = tmp_path / name
    target.write_text(payload, encoding="utf-8")
    return target


def test_error_repair_balances_equation_stars(tmp_path):
    tex_path = _write(
        tmp_path,
        "main.tex",
        "\\begin{equation*} a = b \\end{equation*}\n\\end{equation*}",
    )
    log_path = _write(tmp_path, "main.log", "Bad math environment delimiter")
    repair = LaTeXErrorRepair()

    assert repair.repair(tex_path, log_path)
    patched = tex_path.read_text(encoding="utf-8")
    assert "\\begin{equation}" in patched
    assert patched.count("\\end{equation}") >= 1
    assert "% removed stray" in patched


def test_error_repair_closes_unbalanced_equations(tmp_path):
    tex_path = _write(tmp_path, "main.tex", "\\begin{equation} x + y = z")
    log_path = _write(tmp_path, "main.log", "Bad math environment delimiter")
    repair = LaTeXErrorRepair()

    assert repair.repair(tex_path, log_path)
    patched = tex_path.read_text(encoding="utf-8")
    assert patched.strip().endswith("\\end{equation}")
