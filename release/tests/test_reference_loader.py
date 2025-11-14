from release.core.reference_loader import load_reference_tex


def test_load_reference_tex_returns_requested_domain(tmp_path):
    root = tmp_path / "reference_tex"
    domain_dir = root / "math"
    domain_dir.mkdir(parents=True)
    (domain_dir / "sample.tex").write_text("\\section{Math}", encoding="utf-8")

    examples = load_reference_tex("math", root=root, max_files=1, max_chars=100)

    assert examples and "Math" in examples[0]


def test_load_reference_tex_falls_back_to_default(tmp_path):
    root = tmp_path / "reference_tex"
    default_dir = root / "default"
    default_dir.mkdir(parents=True)
    (default_dir / "baseline.tex").write_text("\\section{Default}", encoding="utf-8")

    examples = load_reference_tex("nonexistent", root=root, max_files=1, max_chars=100)

    assert examples and "Default" in examples[0]


def test_style_domain_disable_token(tmp_path):
    root = tmp_path / "reference_tex"
    assert load_reference_tex("none", root=root) == []
