from typing import Any, List, Union
import numpy as np
from .base import TextRecognizer

try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PaddleOCR = None
    PADDLE_AVAILABLE = False

class PaddleTextRecognizer(TextRecognizer):
    def __init__(self, use_angle_cls: bool = True, lang: str = 'en', show_log: bool = False):
        # show_log ignored in mock and potentially in new paddleocr
        self.mock_mode = False
        
        if not PADDLE_AVAILABLE:
            print("WARNING: PaddleOCR not installed. Running in MOCK mode.")
            self.mock_mode = True
            self.ocr = None
            return

        try:
            # Initialize PaddleOCR (show_log removed as it caused ValueError in new versions)
            self.ocr = PaddleOCR(use_angle_cls=use_angle_cls, lang=lang)
        except Exception as e:
            print(f"WARNING: Failed to initialize PaddleOCR ({e}). Running in MOCK mode.")
            self.mock_mode = True
            self.ocr = None

    def recognize(self, image: Union[str, np.ndarray], lang: str = "en") -> str:
        """
        Recognize text from an image.
        """
        if self.mock_mode:
            return "Mock Text Block"

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