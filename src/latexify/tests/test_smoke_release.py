import argparse
from pathlib import Path

import pytest

import run_release
from latexify.pipeline import rag


@pytest.mark.slow
def test_smoke_pipeline_produces_rewards(tmp_path):
    pdf_path = Path("src/latexify/samples/sample.pdf")
    if not pdf_path.exists():
        pytest.skip("sample PDF missing")
    rag_source = Path("release/reference_tex")
    rag_cache = tmp_path / "cache" / "rag_index.json"
    rag_cache.parent.mkdir(parents=True, exist_ok=True)
    if rag_source.exists():
        rag.build_index(rag_source, rag_cache)
    args = argparse.Namespace(
        pdf=str(pdf_path),
        title=None,
        author="SmokeTest",
        run_dir=str(tmp_path / "smoke_run"),
        chunk_chars=800,
        skip_compile=True,
        log_level="ERROR",
        benchmark_dir=None,
        benchmark_limit=1,
        rag_cache=str(rag_cache),
        rag_cache_budget_mb=0,
        rag_refresh=False,
        reward_mode="heuristic",
        llm_mode="off",
        llm_repo=None,
    )
    tex_path = run_release.run_pipeline(args)
    rewards_path = tex_path.parent / "reports" / "rewards.json"
    assert rewards_path.exists()
