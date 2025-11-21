from latexify.pipeline import assembly
from latexify.core import common
from latexify.core.sanitizer import sanitize_unicode_to_latex


def test_unicode_sanitizer_rewrites_common_symbols():
    payload = "x ∈ A ∪ B ⇒ x ≥ 0 and y ≤ ∞ − 1′"
    sanitized = sanitize_unicode_to_latex(payload)

    assert "∈" not in sanitized
    assert r"\in" in sanitized
    assert r"\cup" in sanitized
    assert r"\geq" in sanitized
    assert "- 1" in sanitized
    assert r"^{\prime}" in sanitized
    assert sanitize_unicode_to_latex("∉") == r"\notin"
    assert "̸" not in sanitize_unicode_to_latex("A̸B")
    assert "?" == sanitize_unicode_to_latex("�")


def test_assembly_stage_applies_sanitizer(tmp_path):
    plan = [
        common.PlanBlock(block_id="b1", chunk_id="c1", label="Intro", block_type="section"),
    ]
    snippets = [
        common.Snippet(chunk_id="c1", latex="Velocity ∈ ℝ^3 ⇒ ∇·u = 0 and τ ≤ 5 − 2′"),
    ]
    plan_path = tmp_path / "plan.json"
    snippets_path = tmp_path / "snippets.json"
    common.save_plan(plan, plan_path)
    common.save_snippets(snippets, snippets_path)
    output_dir = tmp_path / "out"

    tex_path = assembly.run_assembly(
        plan_path=plan_path,
        snippets_path=snippets_path,
        preamble_path=None,
        output_dir=output_dir,
        title="Test",
        author="Unit",
        skip_compile=True,
    )
    tex_text = tex_path.read_text(encoding="utf-8")
    assert "∈" not in tex_text
    assert r"\in" in tex_text
    assert r"\leq" in tex_text

    # Disabling the sanitizer should preserve the raw glyphs for debugging.
    raw_dir = tmp_path / "raw"
    raw_tex = assembly.run_assembly(
        plan_path=plan_path,
        snippets_path=snippets_path,
        preamble_path=None,
        output_dir=raw_dir,
        title="Test",
        author="Unit",
        skip_compile=True,
        use_unicode_sanitizer=False,
    ).read_text(encoding="utf-8")
    assert "∈" in raw_tex
