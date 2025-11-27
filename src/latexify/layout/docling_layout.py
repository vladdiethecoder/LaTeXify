import logging
from pathlib import Path
from typing import List, Dict, Any, Union

try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.pipeline_options import PipelineOptions, PdfPipelineOptions
    from docling.datamodel.base_models import InputFormat
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False

from .base import LayoutEngine

LOGGER = logging.getLogger(__name__)

class DoclingLayout(LayoutEngine):
    def __init__(self, use_vlm: bool = True, vlm_model: str = "granite_docling"):
        if not DOCLING_AVAILABLE:
            raise ImportError("Docling is not installed. Please install with 'pip install docling[vlm]'.")
        
        self.use_vlm = use_vlm
        pipeline_options = PipelineOptions()
        # pipeline_options.do_ocr = True # We might want layout only, but Docling does both. 
        # For layout detection, we just need the boxes.
        
        if use_vlm:
            # Note: Specific VLM configuration depends on Docling version
            # Assuming standard setup for now
            pass
            
        self.converter = DocumentConverter(pipeline_options=pipeline_options)

    def detect(self, image_path: Path) -> List[Dict[str, Any]]:
        """
        Detect layout using Docling on a single image.
        """
        try:
            # Docling can convert images
            doc = self.converter.convert(image_path).document
            
            results = []
            # Iterate over elements in the document
            # Docling structure: doc.pages[0].items...
            # We need to map Docling items to our bbox format
            # Our format (from yolo.py likely): {"bbox": [x1, y1, x2, y2], "label": "text", "score": 1.0}
            
            # Note: Docling coordinates might need normalization or checking
            for item in doc.texts: # accessing text items
                # item.prov gives provenance (bbox on page)
                # item.label gives type (paragraph, heading, etc.)
                for prov in item.prov:
                    bbox = [prov.bbox.l, prov.bbox.t, prov.bbox.r, prov.bbox.b]
                    results.append({
                        "bbox": bbox,
                        "label": item.label.value if hasattr(item.label, 'value') else str(item.label),
                        "score": 1.0, # Docling doesn't always give a score per item
                        "text": item.text # Optional: keep text if available
                    })
            
            for item in doc.tables:
                for prov in item.prov:
                    bbox = [prov.bbox.l, prov.bbox.t, prov.bbox.r, prov.bbox.b]
                    results.append({
                        "bbox": bbox,
                        "label": "table",
                        "score": 1.0
                    })
                    
            for item in doc.pictures:
                for prov in item.prov:
                    bbox = [prov.bbox.l, prov.bbox.t, prov.bbox.r, prov.bbox.b]
                    results.append({
                        "bbox": bbox,
                        "label": "figure",
                        "score": 1.0
                    })

            # Formulas?
            # Docling might group them under texts or specific formula items depending on version
            
            return results

        except Exception as e:
            LOGGER.error(f"Docling layout failed for {image_path}: {e}")
            return []
