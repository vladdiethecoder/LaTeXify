import argparse
import asyncio
import logging
from pathlib import Path
import sys

from latexify.core.state import DocumentState
from latexify.core.pipeline import PipelineRunner
from latexify.steps.pdf_reader import PDFReaderStep
from latexify.steps.layout_analysis import LayoutAnalysisStep
from latexify.steps.ocr import OCRStep
from latexify.steps.chunking import ChunkingStep

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    parser = argparse.ArgumentParser(description="Run LaTeXify Pipeline")
    parser.add_argument("pdf_path", type=Path, help="Path to input PDF")
    parser.add_argument("--ocr-mode", default="auto", help="OCR backend mode")
    args = parser.parse_args()

    if not args.pdf_path.exists():
        logger.error(f"File not found: {args.pdf_path}")
        sys.exit(1)

    # Initialize State
    doc_id = args.pdf_path.stem
    initial_state = DocumentState(
        doc_id=doc_id,
        file_path=args.pdf_path
    )

    # Configure Steps
    steps = [
        PDFReaderStep(),
        LayoutAnalysisStep(),
        OCRStep(mode=args.ocr_mode),
        ChunkingStep()
    ]

    # Run Pipeline
    runner = PipelineRunner(steps)
    logger.info(f"Starting pipeline for {doc_id}...")
    
    try:
        final_state = await runner.run(initial_state)
        logger.info("Pipeline completed successfully.")
        
        # Output summary
        print(f"--- Summary for {final_state.doc_id} ---")
        print(f"Status: {final_state.status}")
        print(f"Raw Pages: {len(final_state.pages)}")
        print(f"Layout Regions: {len(final_state.ocr_blocks)}")
        print(f"OCR Content Pages: {len(final_state.ocr_content)}")
        print(f"Chunks: {len(final_state.chunks)}")
        
        # Save chunks for inspection
        output_path = args.pdf_path.parent / f"{doc_id}_chunks.json"
        import json
        output_path.write_text(json.dumps(final_state.chunks, indent=2, default=str))
        print(f"Chunks saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
