import json
from scripts.latex_deps_check import parse_missing_styles, suggest_installs

def test_gs1_detection_and_suggestion():
    fake_log = """
    ! LaTeX Error: File `GS1.sty' not found.
    See the LaTeX manual or LaTeX Companion for explanation.
    """
    missing = parse_missing_styles(fake_log)
    assert "GS1.sty" in missing

    sugg = suggest_installs(missing)
    dnf = " ".join(sugg["dnf"])
    tlmgr = " ".join(sugg["tlmgr"])
    assert "texlive-gs1" in dnf
    assert "tlmgr install gs1" in tlmgr
