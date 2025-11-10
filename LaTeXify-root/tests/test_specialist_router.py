from __future__ import annotations

from latexify.pipeline.specialist_router import SpecialistRouter


def test_router_heuristic_respects_weights(tmp_path):
    cfg = tmp_path / "router.yaml"
    cfg.write_text(
        "\n".join(
            [
                "weights:",
                "  math: 2.0",
            ]
        ),
        encoding="utf-8",
    )
    plan = {"tasks": [{"id": "Q1", "kind": "math"}]}
    router = SpecialistRouter(plan=plan, config_path=cfg)
    bundle = {"task_id": "Q1", "content_type": "math"}
    decision = router.route(bundle, plan_info={"kind": "math"})
    assert decision.name == "math"
    assert decision.metadata["router_source"] in {"heuristic", "override"}


def test_router_task_override(tmp_path):
    cfg = tmp_path / "router.yaml"
    cfg.write_text(
        "\n".join(
            [
                "tag_overrides:",
                "  task_ids:",
                "    FIG1: figure",
            ]
        ),
        encoding="utf-8",
    )
    plan = {"tasks": [{"id": "FIG1", "kind": "text"}]}
    router = SpecialistRouter(plan=plan, config_path=cfg)
    decision = router.route({"task_id": "FIG1"}, plan_info={})
    assert decision.name == "figure"
    assert decision.metadata["router_source"] == "override"
