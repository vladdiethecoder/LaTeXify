from pathlib import Path

import pytest

from latexify.scripts import bootstrap_env


def test_venv_bin_posix(tmp_path, monkeypatch):
    monkeypatch.setattr(bootstrap_env, "os", bootstrap_env.os)
    monkeypatch.setattr(bootstrap_env.os, "name", "posix")
    venv_path = tmp_path / ".venv"
    expected = venv_path / "bin" / "python"
    assert bootstrap_env._venv_bin(venv_path, "python") == expected


def test_venv_bin_windows(monkeypatch, tmp_path):
    class FakeOS:
        name = "nt"

    monkeypatch.setattr(bootstrap_env, "os", FakeOS())
    venv_path = tmp_path / ".venv"
    assert bootstrap_env._venv_bin(venv_path, "python").as_posix().endswith("Scripts/python.exe")


def test_locate_installer_finds_primary(monkeypatch, tmp_path):
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir(parents=True)
    installer = scripts_dir / "install_models.py"
    installer.write_text("MODEL_REGISTRY = {}", encoding="utf-8")
    monkeypatch.setattr(bootstrap_env, "REPO_ROOT", tmp_path)
    assert bootstrap_env._locate_installer() == installer


def test_locate_installer_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(bootstrap_env, "REPO_ROOT", tmp_path)
    with pytest.raises(FileNotFoundError):
        bootstrap_env._locate_installer()


def test_load_installer_module(monkeypatch, tmp_path):
    scripts_dir = tmp_path / "release" / "scripts"
    scripts_dir.mkdir(parents=True)
    installer = scripts_dir / "install_models.py"
    installer.write_text("value = 42", encoding="utf-8")
    monkeypatch.setattr(bootstrap_env, "REPO_ROOT", tmp_path)
    module = bootstrap_env._load_installer_module()
    assert getattr(module, "value", None) == 42
