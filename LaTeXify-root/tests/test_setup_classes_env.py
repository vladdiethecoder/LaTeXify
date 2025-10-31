import os, sys, json, pathlib
from pathlib import Path

def test_write_latexmkrc_token_replacement(tmp_path: Path, monkeypatch):
    proj = tmp_path
    (proj / "kb" / "online" / "github" / "LiX-master").mkdir(parents=True)
    (proj / "kb" / "online" / "github" / "LiX-master" / "lix.sty").write_text("%", encoding="utf-8")

    sys.path.insert(0, str(Path.cwd()))
    from scripts.setup_classes import write_latexmkrc, stage_lix

    log = proj / "build" / "setup_classes.log.jsonl"
    texmf = proj / "kb" / "classes"
    staged = stage_lix(proj / "kb" / "online" / "github" / "LiX-master", texmf, log)
    rc = write_latexmkrc(proj / "build", texmf, log)

    s = Path(rc).read_text(encoding="utf-8")
    assert "TEXMFHOME" in s and "__TEXMFHOME__" not in s
    # Ensure braces survived literally (we had a KeyError before)
    assert "$ENV{'TEXMFHOME'}" in s
    assert str(texmf.resolve()).replace("\\", "/") in s
