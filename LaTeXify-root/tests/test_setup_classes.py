from pathlib import Path
from scripts.setup_classes import stage_lix

def test_stage_lix_minimal(tmp_path: Path):
    # fake LiX repo
    repo = tmp_path / "LiX"
    (repo / "classes").mkdir(parents=True, exist_ok=True)
    (repo / "lix.sty").write_text("% dummy", encoding="utf-8")
    (repo / "classes" / "textbook.cls").write_text("% dummy", encoding="utf-8")

    log = tmp_path / "log.jsonl"
    texmf = tmp_path / "texmf"

    staged, issues, out_dir = stage_lix(repo, texmf, log)
    assert not issues
    names = [Path(s["dst"]).name for s in staged]
    assert "lix.sty" in names and "textbook.cls" in names
