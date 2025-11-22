import sys
import argparse
from pathlib import Path
from omegaconf import OmegaConf
import logging
import shutil

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
        
    if args.qa_model:
        if "qa" not in cfg.model:
            cfg.model.qa = {}
        cfg.model.qa.repo_id = args.qa_model
        
    if args.qa_device:
        if "hardware" not in cfg:
            cfg.hardware = {}
        cfg.hardware.qa_device = args.qa_device
        
    return cfg

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
    pipeline = LaTeXifyPipeline(cfg)
    
    logger.info(f"Processing {input_pdf}...")
    try:
        result = pipeline.process(input_pdf)
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
    
    # Ignored args for now but accepted to prevent crashing if user passes them
    parser.add_argument("--qa-threshold", type=float, help="QA Threshold")
    parser.add_argument("--max-reruns", type=int, help="Max reruns")
    parser.add_argument("--rerun-delay", type=str, help="Rerun delay")
    parser.add_argument("--log-level", type=str, help="Log level")

    args = parser.parse_args()
    run_pipeline(args)

if __name__ == "__main__":
    main()
