from abc import ABC, abstractmethod
from typing import Any, Optional

class MathRecognizer(ABC):
    @abstractmethod
    def predict(self, image: Any) -> str:
        """
        Predict LaTeX from an equation image.
        
        Args:
            image: PIL Image or numpy array.
            
        Returns:
            str: LaTeX string.
        """
        pass
