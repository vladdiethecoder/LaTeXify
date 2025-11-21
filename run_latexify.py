import sys
from pathlib import Path
from omegaconf import OmegaConf
import logging

# Add src to sys.path
sys.path.append(str(Path(__file__).parent / "src"))

from latexify.core.pipeline import LaTeXifyPipeline
from latexify.core.compiler import LatexCompiler

def load_config():
    # Manual config loading to bypass Hydra/Python 3.14 incompatibility
    conf_dir = Path("config")
    pipeline_cfg = OmegaConf.load(conf_dir / "pipeline.yaml")
    
    # Load defaults manually (simplified)
    if "defaults" in pipeline_cfg:
        # We know the structure: hardware=rtx5090, model=unimer
        # Load hardware
        hw_cfg = OmegaConf.load(conf_dir / "hardware" / "rtx5090.yaml")
        # Load model
        model_cfg = OmegaConf.load(conf_dir / "model" / "unimer.yaml")
        
        # Merge: pipeline overrides model/hardware defaults? 
        # Usually defaults are base.
        # Structure in pipeline.yaml suggests:
        # pipeline: ...
        
        # Actually pipeline.yaml has `defaults` list.
        # We will just build a composite config.
        cfg = OmegaConf.merge(
            OmegaConf.create({"pipeline": {"hardware": hw_cfg, "model": model_cfg}}),
            pipeline_cfg
        )
        return cfg
    return pipeline_cfg

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Loading configuration...")
    cfg = load_config()
    
    # For now, hardcoding input pdf
    input_pdf = "src/latexify/inputs/sample.pdf" 
    
    logger.info("Initializing Pipeline...")
    pipeline = LaTeXifyPipeline(cfg)
    
    logger.info("Running Pipeline...")
    result = pipeline.process(Path(input_pdf))
    
    output_path = Path("output.tex")
    output_path.write_text(result, encoding="utf-8")
    logger.info(f"Output written to {output_path}")
    
    logger.info("Compiling to PDF...")
    compiler = LatexCompiler(engine="tectonic")
    success, log = compiler.compile(result, output_pdf_path=Path("sample.pdf"))
    if success:
        logger.info(f"PDF compiled successfully: {Path('sample.pdf').absolute()}")
    else:
        logger.error(f"PDF Compilation failed: {log}")

if __name__ == "__main__":
    main()