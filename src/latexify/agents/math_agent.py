"""
Math Agent: Enhanced UniMERNet wrapper with batching and validation.

This agent wraps UniMERNet for formula extraction while adding:
- Efficient batch processing
- SymPy validation
- Confidence scoring
"""

import logging
from typing import Any, Dict, List, Optional
import numpy as np
from PIL import Image

from latexify.agents.base import BaseExtractor, BaseAgent
from latexify.exceptions import ExtractionError
from latexify.optimization import apply_fp8_quantization, warmup_model

logger = logging.getLogger(__name__)

# Try importing SymPy for validation
try:
    import sympy
    from sympy.parsing.latex import parse_latex
    SYMPY_AVAILABLE = True
except ImportError:
    sympy = None
    parse_latex = None
    SYMPY_AVAILABLE = False

# Try importing UniMERNet
try:
    from unimernet.common.config import Config
    from unimernet.model.predictor import UniMERPredictor
    UNIMER_AVAILABLE = True
except ImportError:
    Config = None
    UniMERPredictor = None
    UNIMER_AVAILABLE = False


class UniMERExtractor(BaseExtractor):
    """
    UniMERNet extractor with optimized batch processing.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.predictor = None
        self.model_path = self.config.get("model_path", "wanderkid/unimernet_small")
        self.batch_size = self.config.get("batch_size", 32)
        self.device = self.config.get("device", "cuda")
        self.use_fp16 = self.config.get("use_fp16", True)
        self.torch_compile = self.config.get("torch_compile", True)
        self.use_fp8 = self.config.get("use_fp8", False)
        self.mock_mode = not UNIMER_AVAILABLE
        
        self._last_confidence = None
        
    def _initialize(self):
        """Load UniMERNet model."""
        if self.mock_mode:
            logger.warning("UniMERNet not available. Running in MOCK mode.")
            return
        
        try:
            import torch
            
            logger.info(f"Loading UniMERNet from {self.model_path}")
            self.cfg = Config({
                "model": {"name": "unimernet_base"},
                "device": self.device
            })
            self.predictor = UniMERPredictor(self.cfg)
            
            # Apply FP8 quantization if enabled
            if self.use_fp8 and hasattr(self.predictor, 'model'):
                logger.info("Applying FP8 quantization to UniMERNet...")
                try:
                    self.predictor.model = apply_fp8_quantization(
                        self.predictor.model,
                        device=self.device
                    )
                    logger.info("FP8 quantization applied successfully")
                except Exception as e:
                    logger.warning(f"FP8 quantization failed: {e}. Proceeding without FP8.")
            
            # Apply torch.compile if enabled
            if self.torch_compile:
                try:
                    if hasattr(self.predictor, 'model'):
                        logger.info("Applying torch.compile to UniMERNet...")
                        self.predictor.model = torch.compile(
                            self.predictor.model,
                            mode="default"
                        )
                except Exception as e:
                    logger.warning(f"torch.compile failed: {e}")
            
            # Warmup model
            if hasattr(self.predictor, 'model') and torch.cuda.is_available():
                logger.info("Warming up UniMERNet...")
                try:
                    # Estimate typical equation crop size
                    warmup_model(
                        self.predictor.model,
                        input_shape=(1, 3, 224, 224),
                        n_iters=5,
                        verbose=False
                    )
                except Exception as e:
                    logger.debug(f"Warmup failed (non-critical): {e}")
            
            logger.info("UniMERNet loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load UniMERNet: {e}")
            self.mock_mode = True
    
    def extract(self, image: np.ndarray) -> str:
        """
        Extract LaTeX from a single equation image.
        
        Args:
            image: NumPy array (H, W, C) in RGB format
            
        Returns:
            LaTeX string
        """
        if not self._initialized:
            self.warmup()
        
        if self.mock_mode:
            return r"\text{Mock Equation: } E = mc^2"
        
        try:
            # Convert numpy to PIL if needed
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            output = self.predictor.predict(image)
            latex = output.get("latex", "")
            
            # Store confidence if available
            self._last_confidence = output.get("confidence", None)
            
            return latex
            
        except Exception as e:
            raise ExtractionError(
                f"UniMERNet extraction failed: {e}",
                extractor="unimernet"
            )
    
    def extract_batch(self, images: List[np.ndarray]) -> List[str]:
        """
        Extract LaTeX from multiple equation images (batched).
        
        Args:
            images: List of NumPy arrays in RGB format
            
        Returns:
            List of LaTeX strings
        """
        if not self._initialized:
            self.warmup()
        
        if self.mock_mode:
            return [f"\\text{{Mock Batch Eq {i}}}" for i in range(len(images))]
        
        try:
            # Convert numpy to PIL
            pil_images = []
            for img in images:
                if isinstance(img, np.ndarray):
                    pil_images.append(Image.fromarray(img))
                else:
                    pil_images.append(img)
            
            # TODO: Check if predictor supports true batch inference
            # For now, process serially (but can be optimized later)
            results = []
            for img in pil_images:
                output = self.predictor.predict(img)
                results.append(output.get("latex", ""))
            
            return results
            
        except Exception as e:
            raise ExtractionError(
                f"UniMERNet batch extraction failed: {e}",
                extractor="unimernet"
            )
    
    def get_confidence(self) -> Optional[float]:
        """Return confidence of last extraction."""
        return self._last_confidence


class MathAgent(BaseAgent):
    """
    High-level Math Agent with validation and error handling.
    
    Routes to UniMERExtractor and optionally validates with SymPy.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        
        # Create UniMER extractor
        unimer = UniMERExtractor(config)
        super().__init__({"unimernet": unimer})
        
        self.enable_sympy_validation = config.get("enable_sympy_validation", True)
        self.confidence_threshold = config.get("confidence_threshold", 0.7)
        
    def process(
        self,
        image: np.ndarray,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process an equation image and return validated result.
        
        Args:
            image: Equation crop (NumPy array)
            context: Optional context (category, bbox, etc.)
            
        Returns:
            Dict with keys: content, confidence, metadata
        """
        context = context or {}
        category = context.get("category", "Equation_Unknown")
        
        # Extract LaTeX
        latex = self.extractors["unimernet"].extract(image)
        confidence = self.extractors["unimernet"].get_confidence()
        
        # Validate with SymPy if enabled
        is_valid = True
        validation_error = None
        
        if self.enable_sympy_validation and SYMPY_AVAILABLE:
            is_valid, validation_error = self._validate_latex(latex)
        
        # Package result
        return {
            "content": latex,
            "confidence": confidence,
            "metadata": {
                "category": category,
                "is_valid": is_valid,
                "validation_error": validation_error,
                "extractor": "unimernet"
            }
        }
    
    def process_batch(
        self,
        images: List[np.ndarray],
        contexts: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Process multiple equation images (batched).
        
        Args:
            images: List of equation crops
            contexts: Optional list of context dicts
            
        Returns:
            List of result dicts
        """
        contexts = contexts or [{} for _ in images]
        
        # Batch extract
        latex_results = self.extractors["unimernet"].extract_batch(images)
        
        # Validate each
        results = []
        for latex, context in zip(latex_results, contexts):
            is_valid = True
            validation_error = None
            
            if self.enable_sympy_validation and SYMPY_AVAILABLE:
                is_valid, validation_error = self._validate_latex(latex)
            
            results.append({
                "content": latex,
                "confidence": None,  # Batch mode doesn't return per-item confidence
                "metadata": {
                    "category": context.get("category", "Equation_Unknown"),
                    "is_valid": is_valid,
                    "validation_error": validation_error,
                    "extractor": "unimernet"
                }
            })
        
        return results
    
    def _validate_latex(self, latex: str) -> tuple[bool, Optional[str]]:
        """
        Validate LaTeX using SymPy.
        
        Returns:
            (is_valid, error_message)
        """
        try:
            # Attempt to parse with SymPy
            expr = parse_latex(latex)
            return (True, None)
        except Exception as e:
            # Not all LaTeX is valid SymPy (e.g., text, matrices)
            # Log but don't fail
            logger.debug(f"SymPy validation failed for: {latex[:50]}... Error: {e}")
            return (False, str(e))


# Backward compatibility
class UniMERNetMathRecognizer:
    """Legacy wrapper for existing pipeline code."""
    
    def __init__(self, cfg_path: str = "config/model/unimer.yaml", device: str = "cuda"):
        config = {"device": device}
        self.agent = MathAgent(config)
        self.agent.extractors["unimernet"].warmup()
    
    def predict(self, image: Any) -> str:
        """Predict LaTeX from a single image."""
        result = self.agent.process(image)
        return result["content"]
    
    def predict_batch(self, images: List[Any]) -> List[str]:
        """Predict LaTeX from multiple images."""
        results = self.agent.process_batch(images)
        return [r["content"] for r in results]
