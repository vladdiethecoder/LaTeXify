from typing import Any, List, Union
import numpy as np
from .base import TextRecognizer

try:
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None

class PaddleTextRecognizer(TextRecognizer):
    def __init__(self, use_angle_cls: bool = True, lang: str = 'en', show_log: bool = False):
        if PaddleOCR is None:
            raise ImportError("PaddleOCR is not installed. Please install it via `pip install paddleocr`.")
        
        # Initialize PaddleOCR
        self.ocr = PaddleOCR(use_angle_cls=use_angle_cls, lang=lang, show_log=show_log)

    def recognize(self, image: Union[str, np.ndarray], lang: str = "en") -> str:
        """
        Recognize text from an image.
        """
        # PaddleOCR expects path or numpy array
        result = self.ocr.ocr(image, cls=True)
        
        if not result or result[0] is None:
            return ""
            
        # Result structure is a list of lists of [box, (text, score)]
        # We just want the text combined.
        full_text = []
        for line in result:
            for word_info in line:
                text, score = word_info[1]
                full_text.append(text)
                
        return " ".join(full_text)
