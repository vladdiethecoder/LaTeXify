from pathlib import Path

import types

from latexify.scripts import install_models


def test_git_clone_installer(monkeypatch, tmp_path):
    calls = []

    def fake_run(cmd, check):
        calls.append(cmd)
        dest = Path(cmd[-1])
        dest.mkdir(parents=True, exist_ok=True)
        (dest / ".git").mkdir()

    monkeypatch.setattr(install_models, "subprocess", types.SimpleNamespace(run=fake_run))

    spec = install_models.ModelSpec(
        key="test/git-repo",
        repo_id=None,
        target=Path("test/git"),
        git_url="https://github.com/fake/repo.git",
    )
    monkeypatch.setattr(install_models, "MODELS_DIR", tmp_path)
    install_models.install_model(spec, dry_run=False, force=True)

    assert calls and calls[0][0] == "git"
    dest = tmp_path / spec.target
    assert not (dest / ".git").exists()
    info = (dest / "MODEL.json").read_text(encoding="utf-8")
    assert "fake/repo" in info
