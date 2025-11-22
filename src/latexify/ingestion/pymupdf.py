import fitz  # type: ignore
import re
import logging
from pathlib import Path
from typing import List, Optional, Any, Dict

from .base import IngestionEngine
from ..core import common

LOGGER = logging.getLogger(__name__)

# Regex heuristics (shared with MinerU adapter)
QUESTION_RE = re.compile(r"^(question|q|problem)[\s\.]*([0-9]+[a-z]?|\\\(?[" +
"w]+\\)?)", re.IGNORECASE)
ANSWER_RE = re.compile(r"^(answer|solution|proof|lemma|theorem)\b", re.IGNORECASE)
FIGURE_RE = re.compile(r"^(figure|fig)[\s\.]*[0-9]+", re.IGNORECASE)
TABLE_RE = re.compile(r"^(table|tab)[\s\.]*[0-9]+", re.IGNORECASE)

class PyMuPDFIngestor(IngestionEngine):
    def __init__(self, dpi: int = 300):
        self.dpi = dpi

    def process(self, pdf_path: Path, workspace: Path, chunk_chars: int) -> Any:
        """
        Semantic Ingestion Fallback: Use PyMuPDF to extract text and images,
        then apply heuristics to tag chunks.
        """
        LOGGER.info(f"Starting Semantic PyMuPDF processing for {pdf_path}")
        
        image_dir = workspace / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Rasterize (Keep original logic for vision agents)
        image_paths = self.ingest(pdf_path, image_dir)
        
        # 2. Extract Layout & Text
        chunks = []
        doc = fitz.open(pdf_path)
        
        chunk_idx = 0
        current_buffer = []
        current_tag = "text"
        
        for page_idx, page in enumerate(doc):
            # Get blocks: (x0, y0, x1, y1, text, block_no, block_type)
            blocks = page.get_text("blocks")
            
            for block in blocks:
                text = block[4].strip()
                bbox = list(block[:4])
                
                if not text:
                    continue
                
                # Determine Tag
                refined_tag = self._refine_tag(text)
                
                # Decide flush
                is_semantic_break = refined_tag in ["question", "figure", "table", "heading"]
                is_tag_change = refined_tag != current_tag and len(current_buffer) > 0
                
                # Calculate current length
                current_len = sum(len(b["text"]) for b in current_buffer)
                is_full = (current_len + len(text)) > chunk_chars

                if (is_semantic_break or is_tag_change or is_full) and current_buffer:
                    chunks.append(self._create_chunk(current_buffer, current_tag, chunk_idx))
                    chunk_idx += 1
                    current_buffer = []
                    current_tag = refined_tag # Update tag for new buffer
                elif is_semantic_break or is_tag_change:
                     current_tag = refined_tag

                # Add to buffer
                current_buffer.append({
                    "text": text,
                    "bbox": bbox,
                    "page_idx": page_idx,
                    "raw_type": "text" # PyMuPDF doesn't distinguish much
                })
                
        # Final flush
        if current_buffer:
            chunks.append(self._create_chunk(current_buffer, current_tag, chunk_idx))
            
        # Save chunks
        chunks_path = workspace / "chunks.json"
        common.save_chunks(chunks, chunks_path)
        
        from latexify.pipeline.ingestion import IngestionResult
        return IngestionResult(
            chunks_path=chunks_path,
            image_dir=image_dir,
            ocr_dir=workspace / "ocr",
            manifest_path=workspace / "pymupdf_manifest.json"
        )

    def _refine_tag(self, text: str) -> str:
        stripped = text.strip().lower()
        if QUESTION_RE.match(stripped):
            return "question"
        if ANSWER_RE.match(stripped):
            return "answer"
        if FIGURE_RE.match(stripped):
            return "figure"
        if TABLE_RE.match(stripped):
            return "table"
        # Simple heuristic for headings: short, starts with number or all caps?
        # Maybe too risky. Let's stick to explicit markers.
        return "text"

    def _create_chunk(self, buffer: List[Dict], tag: str, idx: int) -> common.Chunk:
        full_text = "\n".join(b["text"] for b in buffer)
        
        bboxes = [b["bbox"] for b in buffer]
        union_bbox = [
            min(b[0] for b in bboxes), min(b[1] for b in bboxes),
            max(b[2] for b in bboxes), max(b[3] for b in bboxes)
        ] if bboxes else None

        metadata = {
            "source": "pymupdf_semantic",
            "tag": tag,
            "bbox": union_bbox,
            "page_start": buffer[0]["page_idx"],
            "page_end": buffer[-1]["page_idx"],
            "contains_images": False, # PyMuPDF 'blocks' usually exclude images unless 'dict' used
            "contains_equations": False # Can't detect easily without OCR
        }
        
        return common.Chunk(
            chunk_id=f"pymupdf_{idx:04d}",
            page=buffer[0]["page_idx"],
            text=full_text,
            metadata=metadata
        )

    def ingest(self, file_path: Path, output_dir: Optional[Path] = None) -> List[Path]:
        """
        Rasterize a PDF into images using PyMuPDF.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        if output_dir is None:
            # Default to a hidden cache dir next to the file
            output_dir = file_path.parent / ".latexify_cache" / file_path.stem
            
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            doc = fitz.open(file_path)
            image_paths = []
            
            for i, page in enumerate(doc):
                # fitz use matrix for dpi. 72 dpi is scale 1.0
                zoom = self.dpi / 72.0
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat)
                
                image_filename = f"page_{{i+1:04d}}.png"
                image_path = output_dir / image_filename
                pix.save(str(image_path))
                image_paths.append(image_path)
                
            return image_paths
        except Exception as e:
            raise RuntimeError(f"Failed to ingest PDF {file_path}: {str(e)}") from e