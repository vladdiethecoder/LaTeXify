"""
Figure Agent: VLM-based figure captioning.

This agent uses a Vision-Language Model (Qwen2.5-VL via vLLM) to generate
descriptive captions for figures extracted from documents.
"""

import logging
from typing import Any, Dict, List, Optional
import numpy as np
from PIL import Image
import base64
from io import BytesIO

from latexify.agents.base import BaseExtractor, BaseAgent
from latexify.exceptions import ExtractionError

logger = logging.getLogger(__name__)


class VLMFigureCaptioner(BaseExtractor):
    """
    VLM-based figure captioning extractor.
    
    Calls vLLM server (or OpenAI-compatible API) for captioning.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.api_base = self.config.get("api_base", "http://localhost:8000/v1")
        self.model_name = self.config.get("model_name", "Qwen/Qwen2.5-VL-72B-Instruct")
        self.temperature = self.config.get("temperature", 0.1)
        self.max_tokens = self.config.get("max_tokens", 300)
        self.prompt = self.config.get(
            "prompt",
            "Describe this figure for a scientific paper. Focus on what is shown, not interpretation. 2-3 sentences."
        )
        
        self.client = None
        self.mock_mode = False
        
    def _initialize(self):
        """Initialize OpenAI client for vLLM API."""
        try:
            from openai import OpenAI
            
            logger.info(f"Connecting to vLLM server at {self.api_base}")
            self.client = OpenAI(
                api_key="EMPTY",  # vLLM doesn't require key
                base_url=self.api_base
            )
            
            # Test connection
            try:
                models = self.client.models.list()
                logger.info(f"Connected to vLLM. Available models: {[m.id for m in models.data]}")
            except Exception as e:
                logger.warning(f"vLLM connection test failed: {e}. Will attempt to use anyway.")
            
        except ImportError:
            logger.error("openai library not installed. Running in MOCK mode.")
            self.mock_mode = True
        except Exception as e:
            logger.error(f"Failed to connect to vLLM: {e}. Running in MOCK mode.")
            self.mock_mode = True
    
    def extract(self, image: np.ndarray) -> str:
        """
        Generate caption for a figure image.
        
        Args:
            image: NumPy array (H, W, C) in RGB format
            
        Returns:
            Caption string
        """
        if not self._initialized:
            self.warmup()
        
        if self.mock_mode:
            return "Mock caption: This figure shows a diagram or chart."
        
        try:
            # Convert image to base64 for API
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image)
            else:
                pil_image = image
            
            # Encode as base64
            buffered = BytesIO()
            pil_image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            img_url = f"data:image/png;base64,{img_str}"
            
            # Call VLM API
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.prompt},
                            {"type": "image_url", "image_url": {"url": img_url}}
                        ]
                    }
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            caption = response.choices[0].message.content.strip()
            logger.debug(f"Generated caption: {caption}")
            
            return caption
            
        except Exception as e:
            raise ExtractionError(
                f"Figure captioning failed: {e}",
                extractor="vlm_captioner"
            )
    
    def extract_batch(self, images: List[np.ndarray]) -> List[str]:
        """
        Generate captions for multiple figures.
        
        Note: Currently processes serially. Could be optimized with async.
        """
        if not self._initialized:
            self.warmup()
        
        if self.mock_mode:
            return [f"Mock caption {i}" for i in range(len(images))]
        
        # Process serially for now
        captions = []
        for img in images:
            try:
                caption = self.extract(img)
                captions.append(caption)
            except Exception as e:
                logger.warning(f"Caption generation failed for image: {e}")
                captions.append("[Caption generation failed]")
        
        return captions


class FigureAgent(BaseAgent):
    """
    High-level Figure Agent for captioning.
    
    Routes to VLMFigureCaptioner.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        
        # Create VLM captioner
        captioner = VLMFigureCaptioner(config)
        super().__init__({"vlm_captioner": captioner})
        
    def process(
        self,
        image: np.ndarray,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a figure image and return caption.
        
        Args:
            image: Figure crop (NumPy array)
            context: Optional context (page_num, etc.)
            
        Returns:
            Dict with keys: content (caption), confidence, metadata
        """
        context = context or {}
        
        # Generate caption
        caption = self.extractors["vlm_captioner"].extract(image)
        
        # Package result
        return {
            "content": caption,
            "confidence": None,  # VLM doesn't provide confidence
            "metadata": {
                "category": "Figure",
                "page_num": context.get("page_num", None),
                "extractor": "vlm_captioner"
            }
        }
    
    def process_batch(
        self,
        images: List[np.ndarray],
        contexts: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Process multiple figures (batched).
        
        Args:
            images: List of figure crops
            contexts: Optional list of context dicts
            
        Returns:
            List of result dicts
        """
        contexts = contexts or [{} for _ in images]
        
        # Batch caption
        captions = self.extractors["vlm_captioner"].extract_batch(images)
        
        # Package results
        results = []
        for caption, context in zip(captions, contexts):
            results.append({
                "content": caption,
                "confidence": None,
                "metadata": {
                    "category": "Figure",
                    "page_num": context.get("page_num", None),
                    "extractor": "vlm_captioner"
                }
            })
        
        return results
