# tests/test_regen_chunks_symlink.py
from __future__ import annotations
import json
from pathlib import Path
import os

def test_regen_unlinks_symlink_and_writes(tmp_path: Path):
    kb = tmp_path / "kb/latex"
    kb.mkdir(parents=True)
    docs = kb / "latex_docs.jsonl"
    docs.write_text(
        "\n".join([
            json.dumps({"title": "align", "question": "How to align?", "answer": "Use amsmath align.", "url":"local://align"}),
            json.dumps({"title": "booktabs", "answer": "Use top/mid/bottom rules.", "url":"local://booktabs"}),
        ]) + "\n",
        encoding="utf-8",
    )

    # Create a dangling symlink at chunks.jsonl pointing outside tmp tree
    (kb / "chunks").mkdir(parents=True)
    symlink_target = Path("/nonexistent/elsewhere/chunks.jsonl")
    try:
        os.symlink(str(symlink_target), kb / "chunks.jsonl")
    except FileExistsError:
        pass

    import scripts.regen_chunks_from_docs as m
    # run with explicit argv so argparse doesn't see pytest's args
    m.main(["--docs", str(docs), "--out_dir", str(kb)])

    # Assert regular files exist and contain rows (not a symlink anymore)
    out1 = kb / "chunks" / "chunks.jsonl"
    out2 = kb / "chunks.jsonl"
    assert out1.exists() and out1.is_file()
    assert out2.exists() and out2.is_file()
    assert not out2.is_symlink()
    assert sum(1 for _ in out1.open("r", encoding="utf-8")) >= 2
