from pathlib import Path
from latexify.agents.class_probe import build_probe_tex

def test_build_probe_tex_includes_class_and_packages(tmp_path: Path):
    tex = build_probe_tex("scrartcl", ["enumitem", "geometry", "microtype"])
    assert "\\documentclass[11pt]{scrartcl}" in tex
    assert "\\usepackage{enumitem}" in tex
    assert "\\usepackage{geometry}" in tex
    assert "\\usepackage{microtype}" in tex
