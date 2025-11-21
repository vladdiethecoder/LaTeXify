from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional

class TextRecognizer(ABC):
    @abstractmethod
    def recognize(self, image: Any, lang: str = "en") -> str:
        """
        Recognize text from an image crop.
        
        Args:
            image: Numpy array or path to image.
            lang: Language code.
            
        Returns:
            str: Recognized text.
        """
        pass
