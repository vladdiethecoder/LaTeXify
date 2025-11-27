import argparse
import sys
from pathlib import Path

# Ensure root is in path for run_release import
sys.path.append(str(Path(__file__).resolve().parents[3]))

import pytest

import run_release
from latexify.pipeline import rag


@pytest.mark.slow
def test_smoke_pipeline_produces_rewards(tmp_path):
    # Prefer smaller PDF for speed in self-improvement loop
    pdf_path = Path("src/latexify/samples/sample_1page.pdf")
    if not pdf_path.exists():
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
        run_dir=str(tmp_path / "smoke_run"), # Note: run_release uses output_dir, checking if we need to alias it
        output_dir=str(tmp_path / "smoke_run"),
        chunk_chars=800,
        skip_compile=False,
        log_level="ERROR",
        benchmark_dir=None,
        benchmark_limit=1,
        rag_cache=str(rag_cache),
        rag_cache_budget_mb=0,
        rag_refresh=False,
        reward_mode="heuristic",
        llm_mode="off",
        llm_repo=None,
        verbose=False,
        # Missing args added below
        pdf_dpi=None,
        ocr_model=None,
        llm_device=None,
        llm_vllm=False,
        disable_refinement=True, # Disable for smoke test to be fast
        qa_model=None,
        qa_device=None,
        qa_vllm=False,
        overwrite=True,
        enable_robust_compilation=None,
        compilation_retry_count=None,
        enable_render_aware=None,
        render_aware_pages=None,
        enable_multi_branch=None,
        branches=None,
        branch_memory_limit=None,
        fusion_strategy=None,
        enable_vision_synthesis=None,
        vision_preset=None,
        layout_backend="docling", # Test the new backend!
        math_ocr_backend="pix2tex",
        qa_threshold=None,
        max_reruns=None,
        rerun_delay=None,
    )
    tex_path = run_release.run_pipeline(args)
    # In full release mode, we expect a PDF, and maybe not rewards.json if benchmarking is off.
    # Let's assert the PDF exists.
    pdf_path_out = tex_path.with_suffix(".pdf")
    assert pdf_path_out.exists()