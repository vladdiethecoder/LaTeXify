#!/usr/bin/env python3
"""
Simple runner for LaTeXify Gen 3.0 with Phase 4 optimizations.

Usage:
    python scripts/run_phase4_pipeline.py src/latexify/samples/sample.pdf
"""

import sys
import logging
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from latexify.core.pipeline import LaTeXifyPipeline
from latexify.core.compiler import LatexCompiler
from omegaconf import OmegaConf

# Try GPU monitoring
try:
    from latexify.utils.gpu_monitor import GPUMonitor
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_phase4_pipeline.py <input.pdf>")
        print(f"Example: python {sys.argv[0]} src/latexify/samples/sample.pdf")
        sys.exit(1)
    
    input_pdf = Path(sys.argv[1])
    if not input_pdf.exists():
        print(f"Error: File not found: {input_pdf}")
        sys.exit(1)
    
    # Setup logging
    output_dir = Path("output") / "phase4_run"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "pipeline.log")
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info("="*80)
    logger.info("LaTeXify Gen 3.0 - Phase 4 Optimized Pipeline")
    logger.info("="*80)
    logger.info(f"Input PDF: {input_pdf}")
    logger.info(f"Output dir: {output_dir}")
    
    # Start GPU monitoring
    if GPU_AVAILABLE:
        gpu_monitor = GPUMonitor()
        logger.info("GPU monitoring enabled")
    
    try:
        # Load config
        logger.info("Loading configuration...")
        conf_dir = Path("config")
        pipeline_cfg = OmegaConf.load(conf_dir / "pipeline.yaml")
        
        try:
            hw_cfg = OmegaConf.load(conf_dir / "hardware" / "rtx5090.yaml")
        except:
            hw_cfg = {}
            logger.warning("Could not load RTX 5090 config, using defaults")
        
        # Merge configs
        cfg = OmegaConf.merge(
            {"hardware": hw_cfg},
            pipeline_cfg
        )
        
        # Log Phase 4 optimizations
        opt_cfg = hw_cfg.get('optimization', {})
        logger.info(f"Phase 4 Optimizations:")
        logger.info(f"  - FP8 Quantization: {opt_cfg.get('use_fp8', False)}")
        logger.info(f"  - torch.compile: {opt_cfg.get('torch_compile', False)}")
        logger.info(f"  - CUDA Graphs: {opt_cfg.get('use_cuda_graphs', False)}")
        logger.info(f"  - FlashAttention-3: {opt_cfg.get('use_flash_attention', False)}")
        
        # Initialize pipeline
        logger.info("Initializing LaTeXify pipeline...")
        start_time = time.time()
        pipeline = LaTeXifyPipeline(cfg)
        init_time = time.time() - start_time
        logger.info(f"Pipeline initialized in {init_time:.2f}s")
        
        # Process PDF
        logger.info(f"Processing {input_pdf.name}...")
        process_start = time.time()
        result = pipeline.process(input_pdf)
        process_time = time.time() - process_start
        logger.info(f"Processing completed in {process_time:.2f}s")
        
        # Write LaTeX output
        output_tex = output_dir / "output.tex"
        output_tex.write_text(result, encoding="utf-8")
        logger.info(f"LaTeX written to: {output_tex}")
        
        # Compile to PDF
        logger.info("Compiling LaTeX to PDF...")
        compiler = LatexCompiler(engine="tectonic")
        output_pdf = output_dir / "output.pdf"
        success, compile_log = compiler.compile(result, output_pdf_path=output_pdf)
        
        if success:
            logger.info(f"✓ PDF compiled successfully: {output_pdf.absolute()}")
        else:
            logger.error(f"✗ PDF Compilation failed:")
            logger.error(compile_log)
        
        # Log timing summary
        logger.info("="*80)
        logger.info("Timing Summary:")
        logger.info(f"  Initialization: {init_time:.2f}s")
        logger.info(f"  Processing: {process_time:.2f}s")
        logger.info(f"  Total: {init_time + process_time:.2f}s")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)
    
    finally:
        # Stop GPU monitor and print stats
        if GPU_AVAILABLE:
            final_stats = gpu_monitor.get_current_stats()
            summary = gpu_monitor.get_stats_summary()
            
            logger.info("="*80)
            logger.info("GPU Statistics:")
            logger.info(f"  Current VRAM: {final_stats.get('vram_used_gb', 0):.2f} GB")
            logger.info(f"  Peak VRAM: {summary.get('peak_vram_gb', 0):.2f} GB")
            logger.info(f"  Temperature: {final_stats.get('temperature_c', 0):.1f} °C")
            if summary.get('num_operations', 0) > 0:
                logger.info(f"  Tracked operations: {summary.get('num_operations', 0)}")
            logger.info("="*80)
        
        logger.info(f"✓ Pipeline complete! Output: {output_dir}")

if __name__ == "__main__":
    main()

