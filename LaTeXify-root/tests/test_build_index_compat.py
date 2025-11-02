# tests/test_build_index_compat.py
from __future__ import annotations
import json
from pathlib import Path
import pytest

faiss = pytest.importorskip("faiss")  # skip if FAISS not present
from latexify.kb.build_index_compat import build_index

def test_build_index_from_single_file(tmp_path: Path):
    run_dir = tmp_path / "kb" / "latex"
    run_dir.mkdir(parents=True, exist_ok=True)
    data = [
        {"id":"a1","text":"align equations with amsmath","page":1,"label":"kb","source_image":"u1","ocr_model":"kb","bbox":[0,0,0,0]},
        {"id":"a2","text":"booktabs tables with siunitx","page":1,"label":"kb","source_image":"u2","ocr_model":"kb","bbox":[0,0,0,0]},
    ]
    (run_dir / "chunks.jsonl").write_text("\n".join(json.dumps(x) for x in data), encoding="utf-8")
    idx, meta = build_index(run_dir)
    assert idx.exists() and meta.exists()
    assert (run_dir / "latex_docs.index").exists()
    assert (run_dir / "latex_docs.meta.json").exists()
