from __future__ import annotations
from pathlib import Path
from latexify.kb.ensure_kb_alias import ensure_kb_alias

def test_ensure_kb_alias_symlink_or_copy(tmp_path: Path):
    offline = tmp_path / "kb" / "offline" / "latex"
    offline.mkdir(parents=True, exist_ok=True)
    (offline / "faiss.index").write_bytes(b"")
    (offline / "faiss.meta.json").write_text('{"dim":32,"size":0,"metas":[]}', encoding="utf-8")

    alias = tmp_path / "kb" / "latex"
    report = ensure_kb_alias(alias, [offline])

    assert report["ok"] is True
    assert (alias / "faiss.index").exists()
    assert (alias / "faiss.meta.json").exists()
    acts = {s["action"] for s in report["staged"]}
    assert acts & {"linked", "copied"}
