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

    def predict_batch(self, images: list[Any]) -> list[str]:
        """
        Predict LaTeX from a batch of equation images.
        Default implementation loops over predict().
        
        Args:
            images: List of PIL Images or numpy arrays.
            
        Returns:
            List[str]: List of LaTeX strings.
        """
        return [self.predict(img) for img in images]
