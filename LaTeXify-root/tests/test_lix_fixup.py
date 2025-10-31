import json
import importlib
import os
import sys
from pathlib import Path

import pytest


def _write_lfs_pointer(p: Path) -> None:
    p.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:deadbeef\n"
        "size 42\n",
        encoding="utf-8",
    )


def test_detect_pointer_and_install_skips_broken_and_weird(monkeypatch, tmp_path, capsys):
    # Simulated TEXMFHOME
    texhome = tmp_path / "texmf"
    monkeypatch.setenv("TEXMFHOME", str(texhome))

    # Fake kpsewhich: returns a LiX class *pointer* inside TEXMFHOME
    fake_cls = texhome / "tex/latex/lix/textbook.cls"
    fake_cls.parent.mkdir(parents=True, exist_ok=True)
    _write_lfs_pointer(fake_cls)

    def fake_run(cmd):
        # scripts.lix_fixup calls a small helper `_run` for kpsewhich
        if cmd[:1] == ["kpsewhich"]:
            if len(cmd) == 2 and cmd[1].endswith(".cls"):
                return 0, str(fake_cls)
            if cmd[:2] == ["kpsewhich", "-var-value=TEXMFHOME"]:
                return 0, str(texhome)
        return 1, ""

    mod = importlib.import_module("scripts.lix_fixup")
    # Monkeypatch low-level runner
    mod._run = lambda c: fake_run(c)

    # Source dir with real files + a broken symlink that should be ignored
    src = tmp_path / "LiX/classes/custom_classes"
    src.mkdir(parents=True, exist_ok=True)
    (src / "textbook.cls").write_text("% real textbook class", encoding="utf-8")
    (src / "lix.sty").write_text("% real lix package", encoding="utf-8")
    try:
        os.symlink("/nonexistent/novel.cls", src / "Link to novel.cls")
    except FileExistsError:
        pass

    # Run the CLI-style entrypoint
    saved = sys.argv[:]
    sys.argv = ["lix_fixup", "--doc_class", "textbook", "--source", str(src), "--install"]
    try:
        # Expect CLI programs to call sys.exit; assert success exit code.
        with pytest.raises(SystemExit) as e:
            mod.main()
        assert e.value.code == 0
    finally:
        sys.argv = saved

    # Verify JSON events were emitted and capture them
    out_lines = [ln for ln in capsys.readouterr().out.strip().splitlines() if ln.strip()]
    events = [json.loads(ln) for ln in out_lines]

    # Sanity: detected pointer, performed install, and post-check is OK/non-pointer
    assert any(ev.get("event") == "lfs_pointer_detected" for ev in events)
    assert any(ev.get("event") == "installed" for ev in events)
    post = [ev for ev in events if ev.get("event") == "post_install_check"][-1]
    assert post["found"] is True and post["is_pointer"] is False
