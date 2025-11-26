import sys
import argparse
import logging
import os
import shutil
from pathlib import Path

from omegaconf import OmegaConf

# Add src to sys.path
sys.path.append(str(Path(__file__).parent / "src"))

from latexify.core.pipeline import LaTeXifyPipeline
from latexify.core.compiler import LatexCompiler

def load_config(args):
    # Manual config loading to bypass Hydra/Python 3.14 incompatibility
    conf_dir = Path("config")
    pipeline_cfg = OmegaConf.load(conf_dir / "pipeline.yaml")
    
    # Load defaults if present
    defaults = OmegaConf.create({})
    if (conf_dir / "hardware" / "rtx5090.yaml").exists():
        hw_cfg = OmegaConf.load(conf_dir / "hardware" / "rtx5090.yaml")
        defaults.merge_with({"hardware": hw_cfg})
    
    if (conf_dir / "model" / "unimer.yaml").exists():
        model_cfg = OmegaConf.load(conf_dir / "model" / "unimer.yaml")
        defaults.merge_with({"model": model_cfg})

    cfg = OmegaConf.merge(defaults, pipeline_cfg)
    
    # Apply CLI Overrides
    if args.pdf_dpi:
        cfg.pipeline.ingestion.dpi = args.pdf_dpi
    
    if args.ocr_model:
        # Assuming ocr model override updates the model name
        cfg.model.name = args.ocr_model
        
    if args.llm_repo:
        # Ensure the key exists even if not in yaml
        if "refinement" not in cfg.pipeline:
            cfg.pipeline.refinement = {}
        cfg.pipeline.refinement.llm_repo = args.llm_repo
        
    if args.llm_device:
        cfg.hardware.llm_device = args.llm_device
        
    if args.llm_vllm:
        cfg.pipeline.refinement.use_vllm = True

    if args.disable_refinement:
        if "refinement" not in cfg.pipeline:
            cfg.pipeline.refinement = {}
        cfg.pipeline.refinement.enabled = False
        
    if args.qa_model:
        if "qa" not in cfg.model:
            cfg.model.qa = {}
        cfg.model.qa.repo_id = args.qa_model
        
    if args.qa_device:
        if "hardware" not in cfg:
            cfg.hardware = {}
        cfg.hardware.qa_device = args.qa_device

    # Optional overrides that map to pipeline runtime toggles
    if args.fusion_strategy:
        if "fusion" not in cfg.pipeline:
            cfg.pipeline.fusion = {}
        cfg.pipeline.fusion.strategy = args.fusion_strategy
    if args.vision_preset:
        if "vision" not in cfg.pipeline:
            cfg.pipeline.vision = {}
        cfg.pipeline.vision.preset = args.vision_preset
    if args.layout_backend:
        if "layout" not in cfg.pipeline:
            cfg.pipeline.layout = {}
        cfg.pipeline.layout.backend = args.layout_backend
    if args.math_ocr_backend:
        if "ocr" not in cfg.pipeline:
            cfg.pipeline.ocr = {}
        cfg.pipeline.ocr.math_backend = args.math_ocr_backend
    
    return cfg

def _apply_runtime_env(args) -> None:
    """Propagate CLI toggles into environment variables consumed downstream."""
    if args.enable_robust_compilation is not None:
        os.environ["LATEXIFY_ENABLE_ROBUST_COMPILATION"] = "1" if args.enable_robust_compilation else "0"
    if args.compilation_retry_count is not None:
        os.environ["LATEXIFY_COMPILATION_RETRY_COUNT"] = str(args.compilation_retry_count)
    if args.enable_render_aware is not None:
        os.environ["LATEXIFY_ENABLE_RENDER_AWARE"] = "1" if args.enable_render_aware else "0"
    if args.render_aware_pages:
        os.environ["LATEXIFY_RENDER_AWARE_PAGES"] = args.render_aware_pages
    if args.enable_multi_branch is not None:
        os.environ["LATEXIFY_ENABLE_MULTI_BRANCH"] = "1" if args.enable_multi_branch else "0"
    if args.branches:
        os.environ["LATEXIFY_BRANCHES"] = args.branches
    if args.branch_memory_limit is not None:
        os.environ["LATEXIFY_BRANCH_MEMORY_LIMIT"] = str(args.branch_memory_limit)
    if args.fusion_strategy:
        os.environ["LATEXIFY_SNIPPET_FUSION_STRATEGY"] = args.fusion_strategy
    if args.enable_vision_synthesis is not None:
        os.environ["LATEXIFY_VISION_SYNTHESIS_ENABLED"] = "1" if args.enable_vision_synthesis else "0"
    if args.vision_preset:
        os.environ["LATEXIFY_VISION_SYNTHESIS_PRESET"] = args.vision_preset
    if args.layout_backend:
        os.environ["LATEXIFY_LAYOUT_BACKEND"] = args.layout_backend
    if args.math_ocr_backend:
        os.environ["LATEXIFY_MATH_OCR_BACKEND"] = args.math_ocr_backend


def run_pipeline(args):
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level)
    logger = logging.getLogger("run_release")
    
    logger.info("Loading configuration...")
    cfg = load_config(args)
    
    input_pdf = Path(args.pdf)
    if not input_pdf.exists():
        logger.error(f"Input file not found: {input_pdf}")
        sys.exit(1)
        
    logger.info("Initializing Pipeline...")
    _apply_runtime_env(args)

    pipeline = LaTeXifyPipeline(cfg)
    
    logger.info(f"Processing {input_pdf}...")
    try:
        result = pipeline.process(input_pdf, skip_compile=args.skip_compile)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)
    
    # Output directory setup
    output_dir = Path(args.output_dir) if args.output_dir else Path("release/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_base = output_dir / input_pdf.stem
    output_tex = output_base.with_suffix(".tex")
    output_pdf = output_base.with_suffix(".pdf")
    
    output_tex.write_text(result, encoding="utf-8")
    logger.info(f"LaTeX written to {output_tex}")
    
    if not getattr(args, "skip_compile", False):
        logger.info("Compiling to PDF...")
        compiler = LatexCompiler(engine="tectonic")
        success, log = compiler.compile(result, output_pdf_path=output_pdf)
        
        if success:
            logger.info(f"PDF compiled successfully: {output_pdf}")
        else:
            logger.error(f"PDF Compilation failed. Log:\n{log}")
            # Don't exit, just report failure so we can inspect tex
            
    return output_tex

def main():
    parser = argparse.ArgumentParser(description="Run LaTeXify Release Pipeline")
    parser.add_argument("--pdf", required=True, help="Path to input PDF")
    parser.add_argument("--pdf-dpi", type=int, help="PDF Rasterization DPI")
    parser.add_argument("--ocr-model", type=str, help="OCR Model Name (e.g. paddle, unimer)")
    parser.add_argument("--llm-repo", type=str, help="Refiner LLM Repo ID")
    parser.add_argument("--llm-device", type=str, help="Device for LLM (cuda:0, cpu)")
    parser.add_argument("--llm-vllm", action="store_true", help="Use vLLM for refinement")
    parser.add_argument("--qa-model", type=str, help="QA Model Repo ID")
    parser.add_argument("--qa-device", type=str, help="Device for QA model")
    parser.add_argument("--qa-vllm", action="store_true", help="Use vLLM for QA")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("--skip-compile", action="store_true", help="Skip PDF compilation")
    parser.add_argument("--disable-refinement", action="store_true", help="Disable refinement stage to avoid LLM downloads")

    # Robust compilation / render-aware controls
    parser.add_argument("--enable-robust-compilation", dest="enable_robust_compilation", action="store_true", help="Enable robust compilation retries")
    parser.add_argument("--disable-robust-compilation", dest="enable_robust_compilation", action="store_false", help="Disable robust compilation retries")
    parser.add_argument("--compilation-retry-count", type=int, help="Number of robust compilation retries")
    parser.add_argument("--enable-render-aware", dest="enable_render_aware", action="store_true", help="Enable render-aware Flux inpainting (requires constraint maps)")
    parser.add_argument("--disable-render-aware", dest="enable_render_aware", action="store_false", help="Disable render-aware Flux inpainting")
    parser.add_argument("--render-aware-pages", type=str, help="Comma-separated page numbers for render-aware processing")

    # Branching / fusion / vision
    parser.add_argument("--enable-multi-branch", dest="enable_multi_branch", action="store_true", help="Enable multi-branch execution")
    parser.add_argument("--disable-multi-branch", dest="enable_multi_branch", action="store_false", help="Disable multi-branch execution")
    parser.add_argument("--branches", type=str, help="Comma-separated branch letters to run (e.g., a,c)")
    parser.add_argument("--branch-memory-limit", type=int, help="VRAM limit in GB per branch")
    parser.add_argument("--fusion-strategy", type=str, help="Snippet fusion strategy (select_best|merge_hybrid|ensemble_average|adaptive|multi_branch)")
    parser.add_argument("--enable-vision-synthesis", dest="enable_vision_synthesis", action="store_true", help="Enable vision synthesis stage")
    parser.add_argument("--disable-vision-synthesis", dest="enable_vision_synthesis", action="store_false", help="Disable vision synthesis stage")
    parser.add_argument("--vision-preset", type=str, help="Vision preset (balanced|fast|quality)")
    parser.add_argument("--layout-backend", type=str, help="Layout backend (yolov10n.pt|surya)")
    parser.add_argument("--math-ocr-backend", type=str, help="Math OCR backend (mathvision|pix2tex|nougat)")
    
    # Ignored args for now but accepted to prevent crashing if user passes them
    parser.add_argument("--qa-threshold", type=float, help="QA Threshold")
    parser.add_argument("--max-reruns", type=int, help="Max reruns")
    parser.add_argument("--rerun-delay", type=str, help="Rerun delay")
    parser.add_argument("--log-level", type=str, help="Log level")

    args = parser.parse_args()
    run_pipeline(args)

if __name__ == "__main__":
    main()
