import logging
from pathlib import Path
from typing import List, Dict, Any

import hydra
from omegaconf import DictConfig
import cv2
from PIL import Image
import fitz

from latexify.ingestion.pymupdf import PyMuPDFIngestor
from latexify.layout.yolo import YOLOLayoutEngine
from latexify.math.unimernet import UniMERNetMathRecognizer
from latexify.ocr.paddle import PaddleTextRecognizer
from latexify.core.reading_order import ReadingOrder, LayoutBlock
from latexify.core.assembler import Assembler
from latexify.refinement.refiner import LLMRefiner
from latexify.core.compiler import LatexCompiler
from latexify.models.donut_adapter import DonutMetadataExtractor

logger = logging.getLogger(__name__)

class LaTeXifyPipeline:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.ingestor = PyMuPDFIngestor(dpi=cfg.pipeline.ingestion.dpi)
        self.layout = YOLOLayoutEngine(model_path=cfg.pipeline.layout.model)
        self.math_ocr = UniMERNetMathRecognizer()
        self.text_ocr = PaddleTextRecognizer(lang=cfg.pipeline.ocr.lang)
        self.reading_order = ReadingOrder()
        self.assembler = Assembler()
        self.compiler = LatexCompiler()
        
        # Optional Metadata Extractor
        self.metadata_extractor = None
        if cfg.pipeline.get("metadata_extraction", False):
             self.metadata_extractor = DonutMetadataExtractor()
        
        if cfg.pipeline.refinement.enabled:
            self.refiner = LLMRefiner(
                use_vllm=cfg.pipeline.refinement.get("use_vllm", True),
                load_in_8bit=cfg.pipeline.refinement.get("load_in_8bit", False),
                load_in_4bit=cfg.pipeline.refinement.get("load_in_4bit", False)
            )
        else:
            self.refiner = None

    def process(self, pdf_path: Path) -> str:
        logger.info(f"Processing {pdf_path}")
        images = self.ingestor.ingest(pdf_path)
        pdf_doc = fitz.open(pdf_path)
        
        full_document_blocks = []
        metadata = {}
        
        # 0. Metadata Extraction (Page 1)
        if self.metadata_extractor and images:
            logger.info("Extracting metadata from Page 1...")
            try:
                metadata = self.metadata_extractor.extract(images[0])
                logger.info(f"Metadata: {metadata}")
            except Exception as e:
                logger.error(f"Metadata extraction failed: {e}")

        for i, img_path in enumerate(images):
            logger.info(f"Analyzing page {i+1}")
            detections = self.layout.detect(img_path)
            
            # 1. Collect all crops first
            img = cv2.imread(str(img_path))
            page_blocks_data = [] # Store (bbox, category, conf, crop)
            
            math_crops = []
            math_indices = []
            
            for j, det in enumerate(detections):
                bbox = det['bbox']
                category = det['class']
                conf = det['confidence']
                
                x1, y1, x2, y2 = map(int, bbox)
                # Clamp to image dimensions
                h, w = img.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 <= x1 or y2 <= y1:
                    continue
                    
                crop = img[y1:y2, x1:x2]
                
                if category in ["Equation_Display", "Equation_Inline"]:
                    crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    math_crops.append(crop_pil)
                    math_indices.append(j)
                    page_blocks_data.append({'bbox': bbox, 'category': category, 'conf': conf, 'content': None})
                else:
                    content = ""
                    if category in ["Text_Block", "Title", "Caption", "List"]:
                        content = self.text_ocr.recognize(crop)
                    elif category == "Table":
                        content = "[TABLE placeholder]"
                    else:
                        content = self.text_ocr.recognize(crop)
                    
                    page_blocks_data.append({'bbox': bbox, 'category': category, 'conf': conf, 'content': content})

            # 2. Run Math Batch
            if math_crops:
                logger.info(f"Batch processing {len(math_crops)} equations...")
                math_results = self.math_ocr.predict_batch(math_crops)
                
                math_ptr = 0
                for item in page_blocks_data:
                    if item['category'] in ["Equation_Display", "Equation_Inline"] and item['content'] is None:
                        item['content'] = math_results[math_ptr]
                        math_ptr += 1

            # Fallback: if no detections produced content, create a whole-page block from PyMuPDF text
            if not page_blocks_data:
                try:
                    page_text = pdf_doc[i].get_text()
                except Exception:
                    page_text = ""
                page_blocks_data.append(
                    {
                        "bbox": [0, 0, img.shape[1], img.shape[0]],
                        "category": "Text_Block",
                        "conf": 1.0,
                        "content": page_text,
                    }
                )

            # 3. Create LayoutBlocks
            page_blocks = []
            for j, item in enumerate(page_blocks_data):
                block = LayoutBlock(
                    id=f"p{i}_b{j}",
                    bbox=item['bbox'],
                    category=item['category'],
                    confidence=item['conf'],
                    page_num=i,
                    content=item['content']
                )
                page_blocks.append(block)
            
            # Sort page blocks
            sorted_blocks = self.reading_order.sort(page_blocks)
            full_document_blocks.extend(sorted_blocks)
            
        # Assemble
        raw_latex = self.assembler.assemble(full_document_blocks, metadata=metadata)
        
        # Debug: Save raw latex
        raw_path = pdf_path.with_name(pdf_path.stem + "_raw.tex")
        try:
            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(raw_latex)
            logger.info(f"Raw LaTeX (pre-refinement) saved to {raw_path}")
        except Exception as e:
            logger.warning(f"Failed to save raw latex: {e}")
        
        # Refine & Compile Loop
        final_latex = raw_latex
        if self.refiner:
            logger.info("Refining LaTeX...")
            # Initial refinement
            final_latex = self.refiner.refine(raw_latex)
            
            # Compilation Loop (Self-Correction)
            MAX_RETRIES = 3
            for attempt in range(MAX_RETRIES):
                logger.info(f"Compilation attempt {attempt+1}/{MAX_RETRIES}")
                success, log = self.compiler.compile(final_latex)
                
                if success:
                    logger.info("Compilation successful!")
                    break
                else:
                    logger.warning(f"Compilation failed. Log snippet: {log[:200]}...")
                    final_latex = self.refiner.fix_error(final_latex, log)
        
        return final_latex
