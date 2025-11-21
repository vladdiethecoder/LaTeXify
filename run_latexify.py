import sys
from pathlib import Path
import hydra
from omegaconf import DictConfig

# Add src to sys.path
sys.path.append(str(Path(__file__).parent / "src"))

from latexify.core.pipeline import LaTeXifyPipeline

@hydra.main(version_base=None, config_path="config", config_name="pipeline")
def main(cfg: DictConfig):
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # TODO: Get input file from cli args correctly with Hydra
    # For now, hardcoding or expecting 'pdf' key in config override
    input_pdf = "src/latexify/inputs/sample.pdf" 
    if "pdf" in cfg:
        input_pdf = cfg.pdf
        
    pipeline = LaTeXifyPipeline(cfg)
    result = pipeline.process(Path(input_pdf))
    
    output_path = Path("output.tex")
    output_path.write_text(result, encoding="utf-8")
    print(f"Output written to {output_path}")

if __name__ == "__main__":
    main()