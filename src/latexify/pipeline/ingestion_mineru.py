"""
MinerU (Magic-PDF) Ingestion Adapter.
Provides SOTA PDF layout analysis and extraction.
"""
from __future__ import annotations

import logging
import json
import os
from pathlib import Path
from typing import Optional, List, Dict, Any

try:
    from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter
    from magic_pdf.pipe.UNIPipe import UNIPipe
    from magic_pdf.config.enums import SupportedPdfParseMethod
except ImportError:
    UNIPipe = None

from latexify.core import common
from latexify.pipeline.semantic_chunking import SemanticChunker

LOGGER = logging.getLogger(__name__)

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
    ) -> Any: # Returns IngestionResult-like object
        
        if not self.is_available():
            raise RuntimeError("MinerU (magic-pdf) is not installed.")

        LOGGER.info(f"Starting MinerU processing for {pdf_path}")
        
        # Prepare output dir
        image_dir = workspace / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Magic-PDF
        try:
            file_content = open(pdf_path, "rb").read()
            # We use auto-detection for method
            jso_useful_key = {"_pdf_type": "", "model_list": []} 
            
            pipe = UNIPipe(file_content, jso_useful_key, image_writer=DiskReaderWriter(str(image_dir)))
            pipe.pipe_classify()
            pipe.pipe_parse()
            
            # Get structured content
            content_list = pipe.get_text_content_list()
            
            # Convert to LaTeXify Chunks
            chunks = self._convert_to_chunks(content_list, chunk_chars)
            
            # Save chunks
            chunks_path = workspace / "chunks.json"
            common.save_chunks(chunks, chunks_path)
            
            # Create pseudo IngestionResult (using a dict or dynamic object to avoid importing IngestionResult if circular)
            # Assuming IngestionResult is simple dataclass
            from latexify.pipeline.ingestion import IngestionResult
            
            return IngestionResult(
                chunks_path=chunks_path,
                image_dir=image_dir,
                ocr_dir=workspace / "ocr", # Placeholder
                manifest_path=workspace / "mineru_manifest.json"
            )
            
        except Exception as e:
            LOGGER.error(f"MinerU processing failed: {e}")
            raise

    def _convert_to_chunks(self, content_list: List[Any], chunk_chars: int) -> List[common.Chunk]:
        chunks = []
        current_text = ""
        chunk_idx = 0
        page_idx = 1 # MinerU might not give strict page breaks in text list easily without inspecting blocks
        
        for item in content_list:
            # item structure depends on magic-pdf version, assuming text blocks
            text = item.get("text", "") if isinstance(item, dict) else str(item)
            
            if len(current_text) + len(text) > chunk_chars:
                chunks.append(common.Chunk(
                    chunk_id=f"mineru_{chunk_idx:04d}",
                    page=page_idx,
                    text=current_text,
                    metadata={"source": "mineru"}
                ))
                current_text = ""
                chunk_idx += 1
            
            current_text += "\n" + text
            
        if current_text:
             chunks.append(common.Chunk(
                chunk_id=f"mineru_{chunk_idx:04d}",
                page=page_idx,
                text=current_text,
                metadata={"source": "mineru"}
            ))
            
        return chunks
