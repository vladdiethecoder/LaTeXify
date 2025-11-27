import logging
import subprocess
import tempfile
import os
from pathlib import Path
from typing import Any
from PIL import Image
import numpy as np

from .base import TextRecognizer

LOGGER = logging.getLogger(__name__)

class NougatOCR(TextRecognizer):
    def __init__(self, model_tag: str = "0.1.0-base", batch_size: int = 1):
        self.model_tag = model_tag
        self.batch_size = batch_size
        # Verify nougat is installed
        try:
            subprocess.run(["nougat", "--help"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
             LOGGER.warning("Nougat CLI not found. Ensure 'nougat-ocr' is installed.")

    def recognize(self, image: Any, lang: str = "en") -> str:
        # Nougat ignores 'lang' as it's specialized for scientific English/Math
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "input.png"
            output_dir = temp_path / "out"
            
            # Save image
            if isinstance(image, (str, Path)):
                img = Image.open(image)
                img.save(input_path)
            elif isinstance(image, np.ndarray):
                img = Image.fromarray(image)
                img.save(input_path)
            elif isinstance(image, Image.Image):
                image.save(input_path)
            else:
                LOGGER.error(f"Unsupported image type: {type(image)}")
                return ""

            # Run Nougat
            # Command: nougat input.png -o output_dir --model model_tag --no-markdown
            cmd = [
                "nougat",
                str(input_path),
                "-o", str(output_dir),
                "--model", self.model_tag,
                "--no-markdown" # Returns raw text/latex mix? PDF says "to get raw Nougat output"
            ]
            
            try:
                subprocess.run(cmd, check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                LOGGER.error(f"Nougat failed: {e.stderr.decode()}")
                return ""
            
            # Read output
            # Nougat outputs [filename].mmd
            output_file = output_dir / "input.mmd"
            if output_file.exists():
                return output_file.read_text(encoding="utf-8")
            else:
                LOGGER.error("Nougat output file not found.")
                return ""
