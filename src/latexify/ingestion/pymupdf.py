import fitz  # type: ignore
from pathlib import Path
from typing import List, Optional
from .base import IngestionEngine

class PyMuPDFIngestor(IngestionEngine):
    def __init__(self, dpi: int = 300):
        """
        Initialize the PyMuPDF ingestor.
        
        Args:
            dpi: Dots Per Inch for rasterization. Default is 300.
        """
        self.dpi = dpi

    def ingest(self, file_path: Path, output_dir: Optional[Path] = None) -> List[Path]:
        """
        Rasterize a PDF into images using PyMuPDF.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        if output_dir is None:
            # Default to a hidden cache dir next to the file
            output_dir = file_path.parent / ".latexify_cache" / file_path.stem
            
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            doc = fitz.open(file_path)
            image_paths = []
            
            for i, page in enumerate(doc):
                # fitz use matrix for dpi. 72 dpi is scale 1.0
                zoom = self.dpi / 72.0
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat)
                
                image_filename = f"page_{i+1:04d}.png"
                image_path = output_dir / image_filename
                pix.save(str(image_path))
                image_paths.append(image_path)
                
            return image_paths
        except Exception as e:
            raise RuntimeError(f"Failed to ingest PDF {file_path}: {str(e)}") from e
