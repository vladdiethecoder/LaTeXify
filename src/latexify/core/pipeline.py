import logging
from pathlib import Path
from typing import List

import hydra
from omegaconf import DictConfig

from latexify.ingestion.pymupdf import PyMuPDFIngestor
from latexify.layout.yolo import YOLOLayoutEngine
from latexify.math.unimernet import UniMERNetMathRecognizer
from latexify.ocr.paddle import PaddleTextRecognizer
from latexify.core.reading_order import ReadingOrder, LayoutBlock
from latexify.core.assembler import Assembler
from latexify.refinement.refiner import LLMRefiner

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
        
        if cfg.pipeline.refinement.enabled:
            self.refiner = LLMRefiner()
        else:
            self.refiner = None

    def process(self, pdf_path: Path) -> str:
        logger.info(f"Processing {pdf_path}")
        images = self.ingestor.ingest(pdf_path)
        
        full_document_blocks = []
        
        for i, img_path in enumerate(images):
            logger.info(f"Analyzing page {i+1}")
            detections = self.layout.detect(img_path)
            
            page_blocks = []
            # Load image for cropping
            # In a real implementation, we'd pass the in-memory image to avoid re-reading
            # But for now, we'll assume we might need to read it again or change the APIs to accept objects
            import cv2
            img = cv2.imread(str(img_path))
            
            for j, det in enumerate(detections):
                bbox = det['bbox']
                category = det['class']
                conf = det['confidence']
                
                # Crop
                x1, y1, x2, y2 = map(int, bbox)
                crop = img[y1:y2, x1:x2]
                
                content = ""
                if category in ["Equation_Display", "Equation_Inline"]:
                    # Convert cv2 to PIL for UniMERNet
                    from PIL import Image
                    crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    content = self.math_ocr.predict(crop_pil)
                elif category in ["Text_Block", "Title", "Caption", "List"]:
                    content = self.text_ocr.recognize(crop)
                elif category == "Table":
                    # TODO: Table recognition
                    content = "[TABLE placeholder]"
                else:
                    # Fallback
                    content = self.text_ocr.recognize(crop)
                    
                block = LayoutBlock(
                    id=f"p{i}_b{j}",
                    bbox=bbox,
                    category=category,
                    confidence=conf,
                    page_num=i,
                    content=content
                )
                page_blocks.append(block)
            
            # Sort page blocks
            sorted_blocks = self.reading_order.sort(page_blocks)
            full_document_blocks.extend(sorted_blocks)
            
        # Assemble
        raw_latex = self.assembler.assemble(full_document_blocks)
        
        # Refine
        if self.refiner:
            logger.info("Refining LaTeX...")
            final_latex = self.refiner.refine(raw_latex)
            return final_latex
        
        return raw_latex