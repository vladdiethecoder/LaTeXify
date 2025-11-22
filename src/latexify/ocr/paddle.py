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
        # Newer PaddleOCR versions route angle classification via initialization args, so drop the old cls kwarg.
        result = self.ocr.ocr(image)

        if not result:
            return ""

        # Newer PaddleOCR returns a dict per page with rec_texts/rec_scores.
        first = result[0]
        if isinstance(first, dict):
            texts = first.get("rec_texts") or []
            if texts:
                return " ".join(texts)

        # Legacy structure: list of lists of [box, (text, score)]
        full_text = []
        for line in result:
            if not line:
                continue
            for word_info in line:
                if isinstance(word_info, (list, tuple)) and len(word_info) >= 2:
                    text = word_info[1][0] if isinstance(word_info[1], (list, tuple)) else word_info[1]
                    full_text.append(text)
        return " ".join(full_text)
