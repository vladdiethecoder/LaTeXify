from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any
# from jaxtyping import Float # commenting out for now as strict typing setup might require more config
# from torch import Tensor

class LayoutEngine(ABC):
    @abstractmethod
    def detect(self, image_path: Path) -> List[Dict[str, Any]]:
        """
        Detect layout elements in an image.
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            List[Dict]: List of detected objects (bbox, class, confidence).
        """
        pass
