import logging
from pathlib import Path
from typing import List, Dict, Any
import fitz # PyMuPDF
from PIL import Image
import numpy as np

from ..core import common
from ..math.pix2tex_model import Pix2TeXModel
# from ..ocr.nougat import NougatOCR # Enable if we want text refinement

LOGGER = logging.getLogger(__name__)

def run_advanced_ocr(pdf_path: Path, chunks: List[common.Chunk]) -> List[common.Chunk]:
    """
    Refine chunks using specialized OCR (pix2tex for math).
    """
    LOGGER.info("Running Advanced OCR (pix2tex)...")
    
    try:
        pix2tex = Pix2TeXModel()
    except ImportError:
        LOGGER.warning("Pix2TeX not available, skipping math refinement.")
        return chunks
    except Exception as e:
        LOGGER.warning(f"Failed to load Pix2TeX: {e}")
        return chunks

    doc = fitz.open(pdf_path)
    updated_chunks = []
    
    for chunk in chunks:
        meta = chunk.metadata or {}
        region_type = meta.get("region_type", "text")
        
        if region_type == "equation" or meta.get("formula_detected"):
            # Extract image
            page_idx = chunk.page - 1 # chunk.page is 1-based usually
            if 0 <= page_idx < len(doc):
                page = doc[page_idx]
                bbox = meta.get("bbox") # [x0, y0, x1, y1]
                
                if bbox:
                    # Docling bbox might need normalization check. 
                    # Assuming Docling coords match PDF coords (usually true).
                    # But Docling might be bottom-up? PyMuPDF is top-down.
                    # Docling v2 usually is top-down.
                    
                    rect = fitz.Rect(bbox)
                    # 2x zoom for better OCR
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=rect)
                    
                    # Convert to PIL
                    img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                    if pix.n == 4: # RGBA
                        img = Image.fromarray(img_data, "RGBA").convert("RGB")
                    else:
                        img = Image.fromarray(img_data, "RGB")
                    
                    latex = pix2tex.predict(img)
                    if latex:
                        LOGGER.info(f"Refined equation chunk {chunk.chunk_id}: {latex[:20]}...")
                        chunk.text = latex
                        chunk.metadata["ocr_engine"] = "pix2tex"
            
        updated_chunks.append(chunk)

    return updated_chunks
