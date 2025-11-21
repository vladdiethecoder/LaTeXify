import sys
from pathlib import Path
from omegaconf import OmegaConf
import logging

# Add src to sys.path
sys.path.append(str(Path(__file__).parent / "src"))

from latexify.core.pipeline import LaTeXifyPipeline

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
    # Fix structure if needed: pipeline.yaml has 'pipeline' key.
    # Hardware/Model seem to be flat.
    # Let's just ensure cfg.pipeline.ingestion exists.
    
    # For now, hardcoding input pdf
    input_pdf = "src/latexify/inputs/sample.pdf" 
    
    logger.info("Initializing Pipeline...")
    pipeline = LaTeXifyPipeline(cfg)
    
    logger.info("Running Pipeline...")
    result = pipeline.process(Path(input_pdf))
    
    output_path = Path("output.tex")
    output_path.write_text(result, encoding="utf-8")
    logger.info(f"Output written to {output_path}")

if __name__ == "__main__":
    main()
