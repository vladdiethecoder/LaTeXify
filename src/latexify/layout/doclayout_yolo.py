"""
DocLayout-YOLO integration for document layout detection.

This module integrates the DocLayout-YOLO model fine-tuned on DocStructBench
for SOTA layout detection on academic documents.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

from ultralytics import YOLO
import torch
import numpy as np

from latexify.agents.base import BaseExtractor
from latexify.core.document_state import BoundingBox, LayoutRegion
from latexify.exceptions import LayoutDetectionError, ModelLoadingError
from latexify.optimization import apply_fp8_quantization, CUDAGraphWrapper, warmup_model

logger = logging.getLogger(__name__)


class DocLayoutYOLODetector(BaseExtractor):
    """
    DocLayout-YOLO detector for document layout analysis.
    
    Supports DocStructBench classes:
    - Text, Title, Header, Footer
    - Table, Figure
    - Equation_Display, Equation_Inline
    - Caption, List, Sidebar
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.model = None
        self.model_path = self.config.get("model_path", "models/doclayout_yolov10.pt")
        self.confidence_threshold = self.config.get("confidence_threshold", 0.5)
        self.iou_threshold = self.config.get("iou_threshold", 0.4)
        self.padding_percent = self.config.get("padding_percent", 0.08)
        self.use_cuda_graphs = self.config.get("use_cuda_graphs", False)
        self.torch_compile = self.config.get("torch_compile", True)
        self.use_fp8 = self.config.get("use_fp8", False)
        self.cuda_graph_wrapper = None
        
    def _initialize(self):
        """Load the YOLO model and optionally apply optimizations."""
        try:
            logger.info(f"Loading DocLayout-YOLO from {self.model_path}")
            self.model = YOLO(self.model_path)
            
            # Apply FP8 quantization if enabled (RTX 5090 optimization)
            if self.use_fp8 and torch.cuda.is_available():
                logger.info("Applying FP8 quantization to DocLayout-YOLO...")
                try:
                    self.model.model = apply_fp8_quantization(
                        self.model.model,
                        device="cuda"
                    )
                    logger.info("FP8 quantization applied successfully")
                except Exception as e:
                    logger.warning(f"FP8 quantization failed: {e}. Proceeding without FP8.")
            
            # Apply torch.compile if enabled (RTX 5090 optimization)
            if self.torch_compile and torch.cuda.is_available():
                logger.info("Applying torch.compile optimization...")
                try:
                    # Note: This may fail on Python 3.14, wrap in try-except
                    self.model.model = torch.compile(
                        self.model.model,
                        mode=self.config.get("compile_mode", "default")
                    )
                except Exception as e:
                    logger.warning(f"torch.compile failed: {e}. Proceeding without compilation.")
            
            # Warmup model
            if torch.cuda.is_available():
                logger.info("Warming up DocLayout-YOLO...")
                warmup_model(
                    self.model.model,
                    input_shape=(1, 3, 640, 640),
                    n_iters=5,
                    verbose=False
                )
            
            logger.info("DocLayout-YOLO loaded successfully")
            
        except Exception as e:
            raise ModelLoadingError(f"Failed to load DocLayout-YOLO: {e}")
    
    def extract(self, image: np.ndarray) -> str:
        """
        Not used for layout detection. Use detect_regions() instead.
        """
        raise NotImplementedError("Use detect_regions() for layout detection")
    
    def detect_regions(
        self,
        image_path: Path,
        page_num: int = 0
    ) -> List[LayoutRegion]:
        """
        Detect layout regions in a document image.
        
        Args:
            image_path: Path to the image file
            page_num: Page number (for region ID generation)
            
        Returns:
            List of LayoutRegion objects with bounding boxes and categories
        """
        if not self._initialized:
            self.warmup()
        
        try:
            # Run YOLO inference
            results = self.model(
                source=str(image_path),
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False
            )
            
            regions = []
            for result in results:
                boxes = result.boxes
                
                for idx, box in enumerate(boxes):
                    # Extract bbox coordinates
                    xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                    x1, y1, x2, y2 = xyxy
                    
                    # Apply smart cropping padding
                    if self.padding_percent > 0:
                        width = x2 - x1
                        height = y2 - y1
                        pad_w = width * self.padding_percent
                        pad_h = height * self.padding_percent
                        
                        x1 = max(0, x1 - pad_w)
                        y1 = max(0, y1 - pad_h)
                        x2 = x2 + pad_w
                        y2 = y2 + pad_h
                    
                    # Get class and confidence
                    cls_id = int(box.cls.item())
                    confidence = float(box.conf.item())
                    category = self.model.names[cls_id]
                    
                    # Create BoundingBox and LayoutRegion
                    bbox = BoundingBox(
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        confidence=confidence
                    )
                    
                    region = LayoutRegion(
                        bbox=bbox,
                        category=category,
                        page_num=page_num,
                        region_id=f"p{page_num}_r{idx}",
                        confidence=confidence,
                        metadata={
                            "model": "doclayout-yolo",
                            "class_id": cls_id
                        }
                    )
                    
                    regions.append(region)
            
            logger.info(f"Detected {len(regions)} regions on page {page_num}")
            return regions
            
        except Exception as e:
            raise LayoutDetectionError(f"Layout detection failed: {e}")
    
    def get_confidence(self) -> Optional[float]:
        """Return average confidence of last detection."""
        # Not applicable for batch detection
        return None


# Backward compatibility wrapper
class YOLOLayoutEngine:
    """
    Legacy wrapper for existing pipeline code.
    Delegates to DocLayoutYOLODetector.
    """
    
    def __init__(self, model_path: str = "yolov10n.pt", config: Optional[Dict[str, Any]] = None):
        # Merge model_path into config
        if config is None:
            config = {}
        config["model_path"] = model_path
        
        self.detector = DocLayoutYOLODetector(config)
        self.detector.warmup()
    
    def detect(self, image_path: Path, page_num: int = 0) -> List[Dict[str, Any]]:
        """
        Detect layout regions (legacy format).
        
        Returns:
            List of dicts with keys: bbox, class, confidence
        """
        regions = self.detector.detect_regions(image_path, page_num)
        
        # Convert to legacy format
        detections = []
        for region in regions:
            detections.append({
                "bbox": [
                    region.bbox.x1,
                    region.bbox.y1,
                    region.bbox.x2,
                    region.bbox.y2
                ],
                "class": region.category,
                "confidence": region.confidence
            })
        
        return detections
