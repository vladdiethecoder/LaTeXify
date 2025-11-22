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

def load_config():
    # Manual config loading to bypass Hydra/Python 3.14 incompatibility
    conf_dir = Path("config")
    pipeline_cfg = OmegaConf.load(conf_dir / "pipeline.yaml")
    
    if "defaults" in pipeline_cfg:
        # Load hardware (assuming rtx5090 default for now or checking config)
        # The pipeline.yaml has defaults list but loading it raw doesn't resolve them.
        # We'll check if the files exist.
        hw_cfg = OmegaConf.load(conf_dir / "hardware" / "rtx5090.yaml")
        model_cfg = OmegaConf.load(conf_dir / "model" / "unimer.yaml")
        
        # Construct the structure expected by pipeline
        # pipeline.yaml usually has 'pipeline' key.
        # We merge defaults into a structure that matches.
        
        defaults = OmegaConf.create({
            "hardware": hw_cfg,
            "model": model_cfg
        })
        
        # Merge pipeline config ON TOP of defaults? 
        # Or usually defaults provide the base, and pipeline.yaml overrides.
        # But pipeline.yaml has the structure 'pipeline: ...'
        
        # Let's just load pipeline.yaml and merge the specific sub-configs 
        # into the places they are expected if they are missing?
        # Actually, LaTeXifyPipeline expects `cfg` to have `pipeline` key, 
        # and probably `hardware` key if it uses it.
        
        # Let's mimic run_latexify.py's logic but clearer
        cfg = OmegaConf.merge(
            defaults,
            pipeline_cfg
        )
        return cfg
    return pipeline_cfg

def run_pipeline(args):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("run_release")
    
    logger.info("Loading configuration...")
    cfg = load_config()
    
    input_pdf = Path(args.pdf)
    if not input_pdf.exists():
        logger.error(f"Input file not found: {input_pdf}")
        sys.exit(1)
        
    logger.info("Initializing Pipeline...")
    pipeline = LaTeXifyPipeline(cfg)
    
    logger.info(f"Processing {input_pdf}...")
    result = pipeline.process(input_pdf)
    
    # Output directory setup
    output_dir = Path(args.run_dir) if hasattr(args, "run_dir") else Path("release/output")
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
            sys.exit(1)
            
    return output_tex

def main():
    parser = argparse.ArgumentParser(description="Run LaTeXify Release Pipeline")
    parser.add_argument("--pdf", required=True, help="Path to input PDF")
    args = parser.parse_args()
    run_pipeline(args)

if __name__ == "__main__":
    main()
