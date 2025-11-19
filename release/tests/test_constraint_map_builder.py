import json

import pytest

pytest.importorskip("reportlab.graphics.shapes")
pytest.importorskip("PIL.Image")

from PIL import Image

from release.pipeline.constraint_map_builder import ConstraintMapBuilder
from release.pipeline.symbolic_render import RenderedFormula


class DummyRenderer:
    def __init__(self, cache_dir):
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._sprite = cache_dir / "dummy_formula.png"
        Image.new("RGBA", (20, 20), (0, 0, 0, 0)).save(self._sprite)

    def render_formula(self, formula: str, fontsize: int = 16) -> RenderedFormula:
        return RenderedFormula(self._sprite, 20, 20, 0.0)


def test_constraint_map_builder_emits_artifacts(tmp_path):
    master_items = [
        {
            "id": "page0001_region000",
            "page": 1,
            "region_type": "formula",
            "bbox": [10.0, 10.0, 110.0, 60.0],
            "polygon": [[10.0, 10.0], [110.0, 10.0], [110.0, 60.0], [10.0, 60.0]],
            "content": "x^2",
            "page_width_pt": 200.0,
            "page_height_pt": 200.0,
        }
    ]
    master_path = tmp_path / "master.json"
    master_path.write_text(json.dumps(master_items), encoding="utf-8")
    renderer = DummyRenderer(tmp_path / "cache")
    builder = ConstraintMapBuilder(renderer, page_images_dir=None)
    artifacts = builder.build_from_master_items(master_path, tmp_path / "out")
    assert artifacts
    artifact = artifacts[0]
    assert artifact.constraint_map.exists(), "Constraint map PNG should be created"
    assert artifact.mask_path.exists(), "Mask PNG should be created"
