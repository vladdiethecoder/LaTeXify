"""
MinerU (Magic-PDF) Ingestion Adapter.
Provides SOTA PDF layout analysis and extraction with Semantic Tagging.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

try:
    from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter
    from magic_pdf.pipe.UNIPipe import UNIPipe
except ImportError:
    UNIPipe = None
    DiskReaderWriter = None

from latexify.core import common
from latexify.pipeline.semantic_chunking import SemanticChunker

LOGGER = logging.getLogger(__name__)

# Heuristics to upgrade MinerU types to Latexify Semantic Types
QUESTION_RE = re.compile(r"^(question|q)[\s\.]*([0-9]+[a-z]?|\\([\w]+\\))", re.IGNORECASE)
ANSWER_RE = re.compile(r"^(answer|solution|proof|lemma|theorem)\b", re.IGNORECASE)

class MinerUIngestion:
    def __init__(self, models_dir: Optional[Path] = None):
        self.models_dir = models_dir
        self._available = UNIPipe is not None

    def is_available(self) -> bool:
        return self._available

    def process(
        self,
        pdf_path: Path,
        workspace: Path,
        chunk_chars: int,
        semantic_chunker: Optional[SemanticChunker] = None
    ) -> Any:
        
        if not self.is_available():
            raise RuntimeError("MinerU (magic-pdf) is not installed.")

        LOGGER.info(f"Starting Semantic MinerU processing for {pdf_path}")
        
        image_dir = workspace / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # 1. Run MinerU Pipeline
            with open(pdf_path, "rb") as f:
                file_content = f.read()
            jso_useful_key = {"_pdf_type": "", "model_list": []}
            
            pipe = UNIPipe(file_content, jso_useful_key, image_writer=DiskReaderWriter(str(image_dir)))
            pipe.pipe_classify()
            pipe.pipe_parse()
            
            # 2. Access the Layout Model (The "Truth" of the document)
            # pipe.model_list contains structured blocks (type, bbox, text)
            if hasattr(pipe, 'model_list'):
                layout_blocks = pipe.model_list
            else:
                # Fallback for different magic-pdf versions
                layout_blocks = pipe.pdf_mid_data.get('model_list', [])

            # 3. Semantic Conversion
            chunks = self._semantic_block_processing(layout_blocks, chunk_chars)
            
            # Save chunks
            chunks_path = workspace / "chunks.json"
            common.save_chunks(chunks, chunks_path)
            
            # Return result pointing to these semantic artifacts
            from latexify.pipeline.ingestion import IngestionResult
            return IngestionResult(
                chunks_path=chunks_path,
                image_dir=image_dir,
                ocr_dir=workspace / "ocr",
                manifest_path=workspace / "mineru_manifest.json"
            )
            
        except Exception as e:
            LOGGER.error(f"MinerU processing failed: {e}", exc_info=True)
            raise

    def _semantic_block_processing(self, blocks: List[Dict], chunk_chars: int) -> List[common.Chunk]:
        """
        Converts MinerU blocks into Latexify chunks, preserving semantic boundaries.
        Merges small adjacent text blocks, but forces breaks on Questions/Figures.
        """
        chunks = []
        current_buffer = []
        current_len = 0
        current_tag = "text"
        chunk_idx = 0
        
        for block in blocks:
            # MinerU types: 'text', 'title', 'interline_equation', 'image', 'table'
            raw_type = block.get('type', 'text')
            text = block.get('text', '') or ""
            # Some versions use 'img_path' or similar for images, logic simplified here
            
            # A. Refine the Tag (Text -> Question | Proof)
            refined_tag = self._refine_tag(raw_type, text)
            
            # B. Decide: Merge or Flush?
            # Flush if:
            # 1. We hit a major semantic boundary (Question, Figure, Table)
            # 2. The new tag is different from the buffer (Header vs Text)
            # 3. Buffer is too full
            
            is_semantic_break = refined_tag in ["question", "figure", "table", "heading"]
            is_tag_change = refined_tag != current_tag and len(current_buffer) > 0
            is_full = (current_len + len(text)) > chunk_chars

            # print(f"DEBUG: Block '{text[:20]}...' | Type: {raw_type} -> {refined_tag} | Current: {current_tag} | Break: {is_semantic_break} | Change: {is_tag_change} | Full: {is_full}")

            if (is_semantic_break or is_tag_change or is_full) and current_buffer:
                # Commit current buffer
                # print(f"DEBUG: Flushing chunk {chunk_idx} (Tag: {current_tag})")
                chunks.append(self._create_chunk(current_buffer, current_tag, chunk_idx))
                chunk_idx += 1
                current_buffer = []
                current_len = 0
            
            # C. Update State
            # If this is a strong tag (Question), it dictates the next buffer's tag
            if is_semantic_break:
                current_tag = refined_tag
            elif is_tag_change:
                # If we switched from Heading -> Text, the new buffer is Text
                current_tag = refined_tag
            
            # D. Add to Buffer
            block_data = {
                "text": text,
                "bbox": block.get("bbox"),
                "raw_type": raw_type,
                "img_path": block.get("img_path"),
                "page_idx": block.get("page_idx", 0)
            }
            current_buffer.append(block_data)
            current_len += len(text)
        
        # Flush remaining
        if current_buffer:
            chunks.append(self._create_chunk(current_buffer, current_tag, chunk_idx))

        return chunks

    def _refine_tag(self, raw_type: str, text: str) -> str:
        """Promotes raw layout types to semantic domain types."""
        if raw_type == "title":
            return "heading"
        if raw_type in ["image", "figure"]:
            return "figure"
        if raw_type == "table":
            return "table"
        if raw_type == "interline_equation":
            return "equation"
        
        # Text analysis for "Question 1", "Solution:", "Proof."
        stripped = text.strip().lower()
        if QUESTION_RE.match(stripped):
            return "question"
        if ANSWER_RE.match(stripped):
            return "answer"
            
        return "text"

    def _create_chunk(self, buffer: List[Dict], tag: str, idx: int) -> common.Chunk:
        """Synthesizes a chunk with rich metadata from the buffer."""
        full_text = "\n".join(b["text"] for b in buffer)
        
        # Calculate aggregate bbox (simplified)
        bboxes = [b["bbox"] for b in buffer if b.get("bbox")]
        union_bbox = [
            min(b[0] for b in bboxes), min(b[1] for b in bboxes),
            max(b[2] for b in bboxes), max(b[3] for b in bboxes)
        ] if bboxes else None

        # Pass specific layout info to metadata so synthesis agents can see it
        metadata = {
            "source": "mineru_semantic",
            "tag": tag,
            "bbox": union_bbox,
            "page_start": buffer[0]["page_idx"],
            "page_end": buffer[-1]["page_idx"],
            "contains_images": any(b.get("raw_type") in ["image", "figure"] for b in buffer),
            "contains_equations": any(b.get("raw_type") == "interline_equation" for b in buffer)
        }
        
        return common.Chunk(
            chunk_id=f"mineru_{idx:04d}",
            page=buffer[0]["page_idx"],
            text=full_text,
            metadata=metadata
        )