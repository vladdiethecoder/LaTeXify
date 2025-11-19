import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("PIL.Image")

from release.pipeline.symbolic_render import FormulaRenderer


def test_formula_renderer_produces_png(tmp_path):
    renderer = FormulaRenderer(tmp_path / "cache")
    result = renderer.render_formula("x^2 + y^2")
    assert result.image_path.exists()
    assert result.width > 0
    assert result.height > 0
