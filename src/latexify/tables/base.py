from abc import ABC, abstractmethod
from typing import Any, Dict

class TableRecognizer(ABC):
    @abstractmethod
    def predict(self, image: Any) -> str:
        """
        Predict LaTeX from a table image.
        
        Args:
            image: PIL Image or numpy array.
            
        Returns:
            str: LaTeX tabular environment string.
        """
        pass
