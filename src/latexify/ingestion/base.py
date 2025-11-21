from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

class IngestionEngine(ABC):
    @abstractmethod
    def ingest(self, file_path: Path, output_dir: Optional[Path] = None) -> List[Path]:
        """
        Ingest a document and return a list of paths to page images.
        
        Args:
            file_path: Path to the input file (PDF).
            output_dir: Optional directory to save the images.
            
        Returns:
            List[Path]: List of paths to the generated images.
        """
        pass
