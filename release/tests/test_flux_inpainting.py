from pathlib import Path

from PIL import Image

from release.pipeline.flux_inpainting import FluxConfig, FluxInpaintingEngine


class DummyPipeline:
    def __init__(self):
        self.called_with: dict[str, object] | None = None

    def __call__(self, **kwargs):
        self.called_with = kwargs
        img = Image.new("RGB", (32, 32), color=(200, 200, 200))
        return type("_Result", (), {"images": [img]})()


def test_flux_engine_runs_with_dummy_pipeline(tmp_path):
    constraint = tmp_path / "constraint.png"
    mask = tmp_path / "mask.png"
    Image.new("RGB", (64, 64), color="white").save(constraint)
    Image.new("L", (64, 64), color=255).save(mask)

    dummy = DummyPipeline()

    def loader(config: FluxConfig):
        return dummy

    engine = FluxInpaintingEngine(FluxConfig(model_id="dummy"), workdir=tmp_path / "renders", pipeline_loader=loader)
    output = engine.generate_page(constraint, mask, page_index=1, steps=5, prompt="test prompt")
    assert output.exists()
    assert dummy.called_with is not None
    assert dummy.called_with["prompt"] == "test prompt"
    assert dummy.called_with["num_inference_steps"] == 5
