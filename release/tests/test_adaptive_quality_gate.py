from release.pipeline.adaptive_quality_gate import (
    AdaptiveQualityGate,
    DocumentProfile,
)


def test_adaptive_gate_passes_with_high_scores(monkeypatch):
    profile = DocumentProfile(doc_type="narrative", math_ratio=0.1, figure_ratio=0.1, difficulty=0.2)
    gate = AdaptiveQualityGate(profile, override=False)
    metrics = {
        "hallucination": {"flagged_count": 0, "total": 5},
        "validation": {"success": True},
        "reward": {"reward": 0.4},
        "cross_validation": {
            "semantic": {"composite": 0.9},
            "content": {"composite": 0.85},
            "structural": {"composite": 0.8},
            "visual": {"composite": 0.9},
        },
    }
    result = gate.evaluate(metrics)
    assert result.passed
    assert result.failed_dimensions == []


def test_adaptive_gate_failure_with_override(monkeypatch):
    profile = DocumentProfile(doc_type="math", math_ratio=0.5, figure_ratio=0.1, difficulty=0.3)
    gate = AdaptiveQualityGate(profile, override=True)
    metrics = {
        "hallucination": {"flagged_count": 2, "total": 2},
        "validation": {"success": False},
        "reward": {"reward": -0.5},
        "cross_validation": {
            "semantic": {"composite": 0.2},
            "content": {"composite": 0.2},
            "structural": {"composite": 0.2},
            "visual": {"composite": 0.2},
        },
    }
    result = gate.evaluate(metrics)
    assert result.passed
    assert result.overrides_applied
