import logging
import sys
from pathlib import Path

# Add src to sys.path
sys.path.append(str(Path(__file__).parent / "src"))

from latexify.config import load_runtime_config
from latexify.core.pipeline import LaTeXifyPipeline
from latexify.core.compiler import LatexCompiler


def main() -> None:
    """Run a simple demo pipeline on the bundled sample PDF."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Loading configuration...")
    runtime_cfg = load_runtime_config("config")

    input_pdf = Path("src/latexify/inputs/sample.pdf")
    if not input_pdf.exists():
        raise FileNotFoundError(input_pdf)

    logger.info("Initializing Pipeline...")
    pipeline = LaTeXifyPipeline(runtime_cfg)

    logger.info("Running Pipeline...")
    result = pipeline.process(input_pdf)

    output_path = Path("output.tex")
    output_path.write_text(result, encoding="utf-8")
    logger.info("Output written to %s", output_path)

    logger.info("Compiling to PDF...")
    compiler = LatexCompiler(engine="tectonic")
    success, log = compiler.compile(result, output_pdf_path=Path("sample.pdf"))
    if success:
        logger.info("PDF compiled successfully: %s", Path("sample.pdf").absolute())
    else:
        logger.error("PDF Compilation failed: %s", log)


if __name__ == "__main__":
    main()
