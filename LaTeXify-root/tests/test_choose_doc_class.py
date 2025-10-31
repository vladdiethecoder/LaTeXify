import json
import subprocess
import sys
from pathlib import Path

def test_choose_doc_class(tmp_path: Path):
    plan = {"doc_class": "lix_article", "tasks": [{"id":"T03","order":3,"title":"Intro","anchor":"intro","depends_on":[]}]}
    p = tmp_path / "plan.json"
    p.write_text(json.dumps(plan), encoding="utf-8")

    out = subprocess.run(
        [sys.executable, "-m", "scripts.choose_doc_class", "--plan", str(p), "--doc_class", "textbook"],
        capture_output=True, text=True, cwd=str(Path.cwd())
    )
    assert out.returncode == 0
    new = json.loads(p.read_text(encoding="utf-8"))
    assert new["doc_class"] == "textbook"
