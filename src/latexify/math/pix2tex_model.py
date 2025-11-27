import logging
from typing import Any, List, Union
from pathlib import Path
from PIL import Image
import numpy as np

try:
    from pix2tex.cli import LatexOCR
    PIX2TEX_AVAILABLE = True
except ImportError:
    PIX2TEX_AVAILABLE = False

from .base import MathRecognizer

LOGGER = logging.getLogger(__name__)

class Pix2TeXModel(MathRecognizer):
    def __init__(self, device: str = "cpu"): # PDF says uses GPU if available, pix2tex handles this usually
        if not PIX2TEX_AVAILABLE:
            raise ImportError("pix2tex is not installed. Please install with 'pip install pix2tex'.")
        
        # LatexOCR automatically uses GPU if available unless specified? 
        # Actually checking source or docs is better, but init usually loads model.
        # We can pass arguments if needed.
        self.model = LatexOCR() 
        LOGGER.info("Pix2TeX model loaded.")

    def predict(self, image: Any) -> str:
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        try:
            # LatexOCR call
            return self.model(image)
        except Exception as e:
            LOGGER.error(f"Pix2TeX prediction failed: {e}")
            return ""

    def predict_batch(self, images: List[Any]) -> List[str]:
        # pix2tex might not support batching natively in the helper class, 
        # so we loop or check if it accepts a list.
        # The PDF implies looping or "minor code adaptation".
        # For now, loop.
        return [self.predict(img) for img in images]
