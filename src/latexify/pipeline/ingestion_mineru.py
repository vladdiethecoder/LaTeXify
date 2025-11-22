"""
MinerU (Magic-PDF 1.x) Ingestion Adapter.
Provides SOTA PDF layout analysis and extraction with Semantic Tagging.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional, List, Dict, Any

try:
    from magic_pdf.data.dataset import PymuDocDataset
    from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
    MAGIC_PDF_AVAILABLE = True
except ImportError:
    MAGIC_PDF_AVAILABLE = False

from latexify.core import common
from latexify.core.state import DocumentState, TextBlock
from latexify.pipeline.semantic_chunking import SemanticChunker
from latexify.ingestion.pymupdf import PyMuPDFIngestor

LOGGER = logging.getLogger(__name__)

# Heuristics to upgrade MinerU types to Latexify Semantic Types
QUESTION_RE = re.compile(r"^(question|q)[\s\.]*([0-9]+[a-z]?|\\([\w]+\\))")
ANSWER_RE = re.compile(r"^(answer|solution|proof|lemma|theorem)\b", re.IGNORECASE)

def ingest_node(state: DocumentState) -> DocumentState:
    """
    Ingestion Node: Runs MinerU to populate layout_blocks AND chunks in DocumentState.
    Falls back to PyMuPDF if MinerU fails (e.g. missing weights).
    """
    if not state.file_path or not state.file_path.exists():
        raise FileNotFoundError(f"Document path not set in state: {state.file_path}")

    LOGGER.info(f"Ingesting {state.file_path}...")
    
    blocks = []
    source = "mineru"

    if MAGIC_PDF_AVAILABLE:
        try:
            LOGGER.info("Attempting MinerU extraction...")
            ingestor = MinerUIngestion()
            blocks = ingestor.extract_blocks(state.file_path)
            LOGGER.info("MinerU extraction successful.")
        except Exception as e:
            LOGGER.warning(f"MinerU extraction failed ({e}). Falling back to PyMuPDF.")
            source = "pymupdf"
    else:
        LOGGER.info("MinerU not available. Using PyMuPDF.")
        source = "pymupdf"
        
    if not blocks and source == "pymupdf":
        # Run PyMuPDF extraction logic here (adapted from PyMuPDFIngestor.process logic)
        # But PyMuPDFIngestor.process returns IngestionResult.
        # We need raw blocks.
        # We can use PyMuPDFIngestor to get blocks if we expose a method or just implement here.
        # Let's use a helper.
        blocks = _extract_pymupdf_blocks(state.file_path)

    # 1. Populate layout_blocks (Raw-ish view)
    state.layout_blocks = [
        TextBlock(
            text=b.get("text", ""),
            bbox=tuple(b.get("bbox")) if b.get("bbox") else None,
            page_index=b.get("page_idx", 0),
            tag=_refine_tag(b.get("category_type") or b.get("type") or "text", b.get("text", "")),
            metadata={"raw_type": b.get("type"), "img_path": b.get("img_path"), "source": source}
        )
        for b in blocks
    ]
    
    # 2. Populate chunks (Semantic view)
    chunk_chars = state.config.get("chunk_chars", 2000)
    state.chunks = _semantic_block_processing(blocks, chunk_chars)
    
    LOGGER.info(f"Ingestion complete ({source}). Found {len(state.layout_blocks)} blocks, merged into {len(state.chunks)} semantic chunks.")
    return state

def _extract_pymupdf_blocks(pdf_path: Path) -> List[Dict]:
    import fitz
    doc = fitz.open(pdf_path)
    blocks = []
    for page_idx, page in enumerate(doc):
        # get_text("blocks") returns (x0, y0, x1, y1, text, block_no, block_type)
        raw_blocks = page.get_text("blocks")
        for b in raw_blocks:
            text = b[4].strip()
            if not text:
                continue
            blocks.append({
                "text": text,
                "bbox": list(b[:4]),
                "page_idx": page_idx,
                "type": "text" # PyMuPDF generic
            })
    return blocks

def _refine_tag(raw_type: str, text: str) -> str:
    raw_type = str(raw_type).lower()
    if raw_type in ["title", "header", "section_header"]:
        return "heading"
    if raw_type in ["image", "figure"]:
        return "figure"
    if raw_type == "table":
        return "table"
    if raw_type in ["interline_equation", "equation", "display_formula"]:
        return "equation"
    
    stripped = text.strip().lower()
    if QUESTION_RE.match(stripped):
        return "question"
    if ANSWER_RE.match(stripped):
        return "answer"
        
    return "text"

def _semantic_block_processing(blocks: List[Dict], chunk_chars: int) -> List[common.Chunk]:
    chunks = []
    current_buffer = []
    current_len = 0
    current_tag = "text"
    chunk_idx = 0
    
    for block in blocks:
        raw_type = block.get('category_type') or block.get('type') or 'text'
        text = block.get('text') or ""
        
        refined_tag = _refine_tag(raw_type, text)
        
        is_semantic_break = refined_tag in ["question", "figure", "table", "heading"]
        is_tag_change = refined_tag != current_tag and len(current_buffer) > 0
        is_full = (current_len + len(text)) > chunk_chars

        if (is_semantic_break or is_tag_change or is_full) and current_buffer:
            chunks.append(_create_chunk(current_buffer, current_tag, chunk_idx))
            chunk_idx += 1
            current_buffer = []
            current_len = 0
        
        if is_semantic_break:
            current_tag = refined_tag
        elif is_tag_change:
            current_tag = refined_tag
        
        block_data = {
            "text": text,
            "bbox": block.get("bbox"),
            "raw_type": raw_type,
            "img_path": block.get("img_path"),
            "page_idx": block.get("page_idx", 0)
        }
        current_buffer.append(block_data)
        current_len += len(text)
    
    if current_buffer:
        chunks.append(_create_chunk(current_buffer, current_tag, chunk_idx))

    return chunks

def _create_chunk(buffer: List[Dict], tag: str, idx: int) -> common.Chunk:
    full_text = "\n".join(b["text"] for b in buffer)
    
    bboxes = [b["bbox"] for b in buffer if b.get("bbox")]
    union_bbox = [
        min(b[0] for b in bboxes), min(b[1] for b in bboxes),
        max(b[2] for b in bboxes), max(b[3] for b in bboxes)
    ] if bboxes else None

    metadata = {
        "source": "mineru_semantic",
        "tag": tag,
        "bbox": union_bbox,
        "page_start": buffer[0]["page_idx"],
        "page_end": buffer[-1]["page_idx"],
        "contains_images": any(b.get("raw_type") in ["image", "figure"] for b in buffer),
        "contains_equations": any(b.get("raw_type") in ["interline_equation", "equation", "display_formula"] for b in buffer)
    }
    
    return common.Chunk(
        chunk_id=f"chunk_{{idx:04d}}",
        page=buffer[0]["page_idx"],
        text=full_text,
        metadata=metadata
    )

class MinerUIngestion:
    def __init__(self, models_dir: Optional[Path] = None):
        self.models_dir = models_dir
        self._available = MAGIC_PDF_AVAILABLE

    def is_available(self) -> bool:
        return self._available

    def extract_blocks(self, pdf_path: Path) -> List[Dict]:
        """Runs MinerU and returns raw layout blocks."""
        if not self.is_available():
            raise RuntimeError("MinerU is not available.")

        with open(pdf_path, "rb") as f:
            file_content = f.read()
        
        ds = PymuDocDataset(file_content)
        infer_result = ds.apply(doc_analyze, ocr=True)
        
        model_list = infer_result.get_infer_res() 
        
        layout_blocks = []
        for page_idx, page_data in enumerate(model_list):
            dets = page_data.get('layout_dets', [])
            for det in dets:
                det['page_idx'] = page_idx
                layout_blocks.append(det)
        
        return layout_blocks

    def process(self, *args, **kwargs):
        # Legacy stub if needed
        pass