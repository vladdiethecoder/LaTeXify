#!/usr/bin/env python3
"""
LaTeXify Gen 3.0 Benchmark Script

Measures end-to-end pipeline performance with Phase 4 optimizations:
- FP8 quantization
- torch.compile()
- Model warmup
- CUDA Graphs (if enabled)

Outputs:
- JSON benchmark report
- Performance breakdown (layout, math, table, assembly)
- VRAM usage stats
- Pages/minute throughput
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from latexify.utils.logging_config import setup_logging
from latexify.utils.gpu_monitor import GPUMonitor

logger = logging.getLogger(__name__)


def run_benchmark(
    pdf_path: Path,
    output_dir: Path,
    config_overrides: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Run full pipeline  benchmark.
    
    Args:
        pdf_path: Input PDF file
        output_dir: Output directory for benchmark results
        config_overrides: Hydra config overrides (e.g., use_fp8=true)
    
    Returns:
        Benchmark results dict
    """
    from latexify.core.pipeline import Pipeline
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize GPU monitor
    gpu_monitor = GPUMonitor(log_interval_s=1.0)
    gpu_monitor.start()
    
    logger.info(f"Starting benchmark on {pdf_path}")
    logger.info(f"Config overrides: {config_overrides}")
    
    # Initialize pipeline
    start_time = time.time()
    pipeline = Pipeline(config_overrides=config_overrides)
    init_time = time.time() - start_time
    
    logger.info(f"Pipeline initialized in {init_time:.2f}s")
    
    # Run pipeline
    try:
        start_time = time.time()
        result = pipeline.run(pdf_path, output_dir)
        total_time = time.time() - start_time
        
        # Stop GPU monitoring
        gpu_monitor.stop()
        gpu_stats = gpu_monitor.get_stats()
        
        # Collect benchmark results
        benchmark = {
            "pdf_path": str(pdf_path),
            "total_time_s": total_time,
            "init_time_s": init_time,
            "pages_processed": result.get("pages_processed", 0),
            "pages_per_minute": (result.get("pages_processed", 0) / total_time) * 60 if total_time > 0 else 0,
            "config": config_overrides,
            "gpu_stats": gpu_stats,
            "stage_times": result.get("stage_times", {}),
            "final_pdf": str(result.get("output_pdf", "")),
            "success": result.get("success", False)
        }
        
        logger.info(f"Benchmark complete: {total_time:.2f}s, {benchmark['pages_per_minute']:.1f} pages/min")
        
        return benchmark
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        gpu_monitor.stop()
        return {
            "pdf_path": str(pdf_path),
            "success": False,
            "error": str(e),
            "config": config_overrides
        }


def compare_configurations(
    pdf_path: Path,
    output_dir: Path
) -> None:
    """
    Run benchmarks with different configurations to compare performance.
    
    Configurations:
    1. Baseline (FP16, no optimizations)
    2. FP8 only
    3. FP8 + torch.compile
    4. Full Phase 4 (FP8 + torch.compile + CUDA Graphs)
    """
    configs = [
        {
            "name": "baseline_fp16",
            "overrides": {"use_fp8": False, "torch_compile": False, "use_cuda_graphs": False}
        },
        {
            "name": "fp8_only",
            "overrides": {"use_fp8": True, "torch_compile": False, "use_cuda_graphs": False}
        },
        {
            "name": "fp8_compile",
            "overrides": {"use_fp8": True, "torch_compile": True, "use_cuda_graphs": False}
        },
        {
            "name": "full_phase4",
            "overrides": {"use_fp8": True, "torch_compile": True, "use_cuda_graphs": True}
        }
    ]
    
    results = []
    
    for config in configs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {config['name']}")
        logger.info(f"{'='*60}")
        
        config_output_dir = output_dir / config["name"]
        benchmark = run_benchmark(pdf_path, config_output_dir, config["overrides"])
        benchmark["config_name"] = config["name"]
        results.append(benchmark)
        
        # Write individual result
        with open(config_output_dir / "benchmark.json", "w") as f:
            json.dump(benchmark, f, indent=2)
    
    # Write comparison report
    comparison = {
        "pdf_path": str(pdf_path),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "configurations": results,
        "speedup_vs_baseline": {}
    }
    
    # Calculate speedups
    baseline_time = next(
        (r["total_time_s"] for r in results if r["config_name"] == "baseline_fp16"),
        None
    )
    
    if baseline_time:
        for result in results:
            if result.get("success") and result["config_name"] != "baseline_fp16":
                speedup = baseline_time / result["total_time_s"]
                comparison["speedup_vs_baseline"][result["config_name"]] = speedup
    
    # Write comparison report
    comparison_path = output_dir / "comparison_report.json"
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)
    
    logger.info(f"\nComparison report saved to: {comparison_path}")
    
    # Print summary table
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    print(f"{'Configuration':<20} {'Time (s)':<12} {'Pages/min':<12} {'Speedup':<10} {'Status'}")
    print("-"*80)
    
    for result in results:
        config_name = result["config_name"]
        if result.get("success"):
            time_s = result["total_time_s"]
            pages_min = result["pages_per_minute"]
            speedup = comparison["speedup_vs_baseline"].get(config_name, 1.0)
            status = "✓ Success"
        else:
            time_s = pages_min = speedup = 0
            status = f"✗ Failed: {result.get('error', 'Unknown')}"
        
        print(f"{config_name:<20} {time_s:<12.2f} {pages_min:<12.1f} {speedup:<10.2f}x {status}")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="LaTeXify Gen 3.0 Benchmark")
    parser.add_argument("pdf_path", type=Path, help="Input PDF file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks") / time.strftime("%Y%m%d_%H%M%S"),
        help="Output directory for benchmark results"
    )
    parser.add_argument(
        "--use-fp8",
        action="store_true",
        help="Enable FP8 quantization"
    )
    parser.add_argument(
        "--torch-compile",
        action="store_true",
        help="Enable torch.compile()"
    )
    parser.add_argument(
        "--compare-configs",
        action="store_true",
        help="Run multiple configurations for comparison"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(
        log_level=logging.DEBUG if args.verbose else logging.INFO,
        log_file=args.output_dir / "benchmark.log"
    )
    
    if not args.pdf_path.exists():
        logger.error(f"PDF not found: {args.pdf_path}")
        sys.exit(1)
    
    if args.compare_configs:
        compare_configurations(args.pdf_path, args.output_dir)
    else:
        # Single run
        config_overrides = {
            "use_fp8": args.use_fp8,
            "torch_compile": args.torch_compile
        }
        
        benchmark = run_benchmark(args.pdf_path, args.output_dir, config_overrides)
        
        # Write result
        with open(args.output_dir / "benchmark.json", "w") as f:
            json.dump(benchmark, f, indent=2)
        
        logger.info(f"Benchmark results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
