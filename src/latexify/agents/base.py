"""
Base abstractions for the LaTeXify Gen 3.0 agent system.

This module defines the core interfaces that all extraction agents must implement,
following the Strategy Pattern for swappable backends.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np
from PIL import Image


class BaseExtractor(ABC):
    """
    Abstract base class for all content extraction agents.
    
    Extraction agents are specialized models that process specific content types
    (text, math, tables, figures) from cropped image regions.
    
    Examples:
        - TextExtractor: Wraps PaddleOCR or Qwen2.5-VL
        - MathExtractor: Wraps UniMERNet
        - TableExtractor: Wraps StructEqTable
        - FigureExtractor: Wraps Qwen2.5-VL for captioning
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the extractor with optional configuration.
        
        Args:
            config: Dictionary of model-specific configuration parameters
        """
        self.config = config or {}
        self._initialized = False
    
    @abstractmethod
    def extract(self, image: np.ndarray) -> str:
        """
        Extract content from a single image crop.
        
        Args:
            image: NumPy array (H, W, C) in RGB format
            
        Returns:
            Extracted content as a string (LaTeX, plain text, etc.)
        """
        pass
    
    def extract_batch(self, images: list[np.ndarray]) -> list[str]:
        """
        Extract content from multiple image crops (batched for efficiency).
        
        Default implementation processes images serially. Subclasses should
        override this for true batched inference.
        
        Args:
            images: List of NumPy arrays in RGB format
            
        Returns:
            List of extracted content strings
        """
        return [self.extract(img) for img in images]
    
    def get_confidence(self) -> Optional[float]:
        """
        Return confidence score of the last extraction (if available).
        
        Returns:
            Float between 0.0 and 1.0, or None if not applicable
        """
        return None
    
    def warmup(self):
        """
        Perform model warmup (load weights, compile, etc.).
        
        Call this once before processing batches to avoid lazy loading overhead.
        """
        if not self._initialized:
            self._initialize()
            self._initialized = True
    
    @abstractmethod
    def _initialize(self):
        """
        Internal initialization logic (load model weights, etc.).
        
        Subclasses must implement this.
        """
        pass


class BaseAgent(ABC):
    """
    Higher-level agent that orchestrates extraction with additional logic.
    
    Agents wrap Extractors and add domain-specific processing:
    - Pre-processing (image enhancement, cropping adjustments)
    - Post-processing (syntax repair, validation)
    - Routing (selecting between multiple extractors)
    
    Examples:
        - MathAgent: Routes to UniMERNet, validates with SymPy
        - TableAgent: Routes to StructEqTable, repairs with LLM
        - TextAgent: Routes between PaddleOCR (fast) and Qwen-VL (handwriting)
    """
    
    def __init__(self, extractors: Dict[str, BaseExtractor]):
        """
        Initialize agent with a dictionary of named extractors.
        
        Args:
            extractors: Dictionary mapping names to BaseExtractor instances
        """
        self.extractors = extractors
    
    @abstractmethod
    def process(self, image: np.ndarray, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process an image crop and return structured results.
        
        Args:
            image: NumPy array (H, W, C) in RGB format
            context: Optional contextual information (bbox, page_num, etc.)
            
        Returns:
            Dictionary with keys: 'content', 'confidence', 'metadata'
        """
        pass
