"""
Table Agent: StructEqTable integration for table structure recognition.

This agent wraps StructEqTable (or similar models) for converting table
images into LaTeX tabular code, with optional LLM repair for syntax errors.
"""

import logging
from typing import Any, Dict, List, Optional
import numpy as np
from PIL import Image

from latexify.agents.base import BaseExtractor, BaseAgent
from latexify.exceptions import ExtractionError

logger = logging.getLogger(__name__)

# Try importing StructEqTable (placeholder - actual library may vary)
try:
    # NOTE: This is a placeholder import
    # Replace with actual StructEqTable library when available
    from structeqtable import TableRecognizer
    STRUCTEQTABLE_AVAILABLE = True
except ImportError:
    TableRecognizer = None
    STRUCTEQTABLE_AVAILABLE = False


class StructEqTableExtractor(BaseExtractor):
    """
    StructEqTable extractor for table structure recognition.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.recognizer = None
        self.model_path = self.config.get("model_path", "models/structeqtable.pt")
        self.max_rows = self.config.get("max_rows", 50)
        self.max_cols = self.config.get("max_cols", 20)
        self.output_format = self.config.get("output_format", "latex")
        self.batch_size = self.config.get("batch_size", 8)
        self.torch_compile = self.config.get("torch_compile", True)
        self.mock_mode = not STRUCTEQTABLE_AVAILABLE
        
    def _initialize(self):
        """Load StructEqTable model."""
        if self.mock_mode:
            logger.warning("StructEqTable not available. Running in MOCK mode.")
            return
        
        try:
            logger.info(f"Loading StructEqTable from {self.model_path}")
            self.recognizer = TableRecognizer(model_path=self.model_path)
            
            # Apply torch.compile if enabled
            if self.torch_compile:
                try:
                    import torch
                    if hasattr(self.recognizer, 'model'):
                        logger.info("Applying torch.compile to StructEqTable...")
                        self.recognizer.model = torch.compile(
                            self.recognizer.model,
                            mode="default"
                        )
                except Exception as e:
                    logger.warning(f"torch.compile failed: {e}")
            
            logger.info("StructEqTable loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load StructEqTable: {e}")
            self.mock_mode = True
    
    def extract(self, image: np.ndarray) -> str:
        """
        Extract LaTeX table from a single table image.
        
        Args:
            image: NumPy array (H, W, C) in RGB format
            
        Returns:
            LaTeX tabular code
        """
        if not self._initialized:
            self.warmup()
        
        if self.mock_mode:
            return self._generate_mock_table()
        
        try:
            # Convert numpy to PIL if needed
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            # Run table recognition
            result = self.recognizer.recognize(image, format=self.output_format)
            latex_table = result.get("latex", "")
            
            return latex_table
            
        except Exception as e:
            raise ExtractionError(
                f"StructEqTable extraction failed: {e}",
                extractor="structeqtable"
            )
    
    def extract_batch(self, images: List[np.ndarray]) -> List[str]:
        """
        Extract LaTeX tables from multiple images (batched).
        
        Args:
            images: List of NumPy arrays in RGB format
            
        Returns:
            List of LaTeX tabular code
        """
        if not self._initialized:
            self.warmup()
        
        if self.mock_mode:
            return [self._generate_mock_table(i) for i in range(len(images))]
        
        try:
            # Convert numpy to PIL
            pil_images = []
            for img in images:
                if isinstance(img, np.ndarray):
                    pil_images.append(Image.fromarray(img))
                else:
                    pil_images.append(img)
            
            # Process serially for now (can be optimized with true batching)
            results = []
            for img in pil_images:
                result = self.recognizer.recognize(img, format=self.output_format)
                results.append(result.get("latex", ""))
            
            return results
            
        except Exception as e:
            raise ExtractionError(
                f"StructEqTable batch extraction failed: {e}",
                extractor="structeqtable"
            )
    
    def _generate_mock_table(self, idx: int = 0) -> str:
        """Generate a mock LaTeX table for testing."""
        return r"""
\begin{tabular}{|c|c|c|}
\hline
Header 1 & Header 2 & Header 3 \\
\hline
Data 1 & Data 2 & Data 3 \\
Data 4 & Data 5 & Data 6 \\
\hline
\end{tabular}
""".strip()


class TableAgent(BaseAgent):
    """
    High-level Table Agent with LLM repair.
    
    Routes to StructEqTableExtractor and optionally repairs syntax with VLM.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        
        # Create StructEqTable extractor
        structeq = StructEqTableExtractor(config)
        super().__init__({"structeqtable": structeq})
        
        self.enable_llm_repair = config.get("enable_llm_repair", True)
        self.vlm_client = None  # Will be injected from pipeline
        self.repair_prompt = config.get(
            "repair_prompt",
            "Fix any LaTeX table syntax errors. Ensure proper alignment, hlines, and cell content. Return only the corrected table code."
        )
    
    def process(
        self,
        image: np.ndarray,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a table image and return validated result.
        
        Args:
            image: Table crop (NumPy array)
            context: Optional context (caption, page_num, etc.)
            
        Returns:
            Dict with keys: content, confidence, metadata
        """
        context = context or {}
        
        # Extract LaTeX
        latex = self.extractors["structeqtable"].extract(image)
        
        # Optionally repair with LLM
        if self.enable_llm_repair and self.vlm_client:
            latex = self._repair_with_llm(latex)
        
        # Look for caption in context
        caption = context.get("caption", None)
        
        # Package result
        return {
            "content": latex,
            "confidence": None,  # StructEqTable doesn't provide confidence
            "metadata": {
                "category": "Table",
                "caption": caption,
                "extractor": "structeqtable",
                "llm_repaired": self.enable_llm_repair and self.vlm_client is not None
            }
        }
    
    def process_batch(
        self,
        images: List[np.ndarray],
        contexts: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Process multiple table images (batched).
        
        Args:
            images: List of table crops
            contexts: Optional list of context dicts
            
        Returns:
            List of result dicts
        """
        contexts = contexts or [{} for _ in images]
        
        # Batch extract
        latex_results = self.extractors["structeqtable"].extract_batch(images)
        
        # Optionally repair each with LLM
        if self.enable_llm_repair and self.vlm_client:
            latex_results = [self._repair_with_llm(latex) for latex in latex_results]
        
        # Package results
        results = []
        for latex, context in zip(latex_results, contexts):
            results.append({
                "content": latex,
                "confidence": None,
                "metadata": {
                    "category": "Table",
                    "caption": context.get("caption", None),
                    "extractor": "structeqtable",
                    "llm_repaired": self.enable_llm_repair and self.vlm_client is not None
                }
            })
        
        return results
    
    def set_vlm_client(self, vlm_client: Any):
        """Inject VLM client for LLM repair."""
        self.vlm_client = vlm_client
    
    def _repair_with_llm(self, latex: str) -> str:
        """
        Repair LaTeX table syntax using VLM.
        
        Args:
            latex: Raw LaTeX table code
            
        Returns:
            Repaired LaTeX table code
        """
        if not self.vlm_client:
            return latex
        
        try:
            # Call VLM for repair (placeholder - actual API depends on client)
            prompt = f"{self.repair_prompt}\n\nTable:\n{latex}"
            repaired = self.vlm_client.generate(prompt)
            
            # Strip markdown code blocks if present
            import re
            code_block_pattern = re.compile(r"```(?:latex)?\s*(.*?)\s*```", re.DOTALL)
            match = code_block_pattern.search(repaired)
            if match:
                repaired = match.group(1).strip()
            
            return repaired
            
        except Exception as e:
            logger.warning(f"LLM repair failed: {e}. Using original table.")
            return latex
