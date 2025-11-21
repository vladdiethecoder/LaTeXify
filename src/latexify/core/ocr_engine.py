"""Refactored OCR Engine using Strategy Pattern."""
from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple, Type

try:
    import torch
except ImportError:
    torch = None

from latexify.config import settings
from latexify.exceptions import OCRError, ModelLoadError
from latexify.core.image_store import PageImageStore
from latexify.core.model_paths import resolve_models_root

LOGGER = logging.getLogger(__name__)

@dataclass
class OCRResult:
    sources: OrderedDict[str, str]
    metadata: Dict[str, Dict[str, object]] = field(default_factory=dict)

    @property
    def backends(self) -> List[str]:
        return list(self.sources.keys())

class OCRBackend(ABC):
    """Abstract base class for OCR backends."""
    
    name: str = "base"
    requires_gpu: bool = False
    
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir

    @abstractmethod
    def is_available(self) -> bool:
        """Check if dependencies and weights are available."""
        pass

    @abstractmethod
    def load(self) -> None:
        """Load model weights."""
        pass

    @abstractmethod
    def unload(self) -> None:
        """Unload model weights to free VRAM."""
        pass

    @abstractmethod
    def process_page(self, image_path: Path) -> Optional[Tuple[str, Dict[str, object]]]:
        """Process a single page image."""
        pass

class NougatBackend(OCRBackend):
    name = "nougat"
    requires_gpu = True

    def __init__(self, models_dir: Path):
        super().__init__(models_dir)
        self._model = None

    def is_available(self) -> bool:
        return (self.models_dir / "ocr" / "nougat-small").exists()

    def load(self) -> None:
        if self._model: return
        try:
            from latexify.models.nougat_adapter import NougatAdapter, NougatAdapterConfig
            model_dir = self.models_dir / "ocr" / "nougat-small"
            self._model = NougatAdapter(NougatAdapterConfig(model_dir=model_dir))
        except Exception as e:
            raise ModelLoadError(f"Failed to load Nougat: {e}")

    def unload(self) -> None:
        self._model = None
        if torch: torch.cuda.empty_cache()

    def process_page(self, image_path: Path) -> Optional[Tuple[str, Dict[str, object]]]:
        if not self._model: self.load()
        try:
            text, conf = self._model.predict_with_confidence(image_path)
            return text, {"confidence": conf}
        except Exception as e:
            LOGGER.warning(f"Nougat failed: {e}")
            return None

class FlorenceBackend(OCRBackend):
    name = "florence2"
    requires_gpu = True

    def __init__(self, models_dir: Path):
        super().__init__(models_dir)
        self._model = None

    def is_available(self) -> bool:
        return (self.models_dir / "ocr" / "florence-2-large").exists()

    def load(self) -> None:
        if self._model: return
        try:
            from latexify.models.model_adapters import FlorenceAdapter, FlorenceConfig
            model_dir = self.models_dir / "ocr" / "florence-2-large"
            device = "cuda" if torch and torch.cuda.is_available() else "cpu"
            self._model = FlorenceAdapter(FlorenceConfig(model_dir=model_dir, device=device))
        except Exception as e:
            raise ModelLoadError(f"Failed to load Florence: {e}")

    def unload(self) -> None:
        if self._model and hasattr(self._model, 'close'):
            self._model.close()
        self._model = None
        if torch: torch.cuda.empty_cache()

    def process_page(self, image_path: Path) -> Optional[Tuple[str, Dict[str, object]]]:
        if not self._model: self.load()
        try:
            text = self._model.predict(image_path)
            return text, {}
        except Exception as e:
            LOGGER.warning(f"Florence failed: {e}")
            return None

class InternVLBackend(OCRBackend):
    name = "internvl"
    requires_gpu = True

    def __init__(self, models_dir: Path):
        super().__init__(models_dir)
        self._model = None
        self._subdir = "ocr/internvl-chat-v1-5" # Simplified path, actual path logic in config

    def is_available(self) -> bool:
        # Simple check, robust check uses resolved path
        return True 

    def load(self) -> None:
        if self._model: return
        try:
            from latexify.models.model_adapters import InternVLAdapter, InternVLConfig
            # Logic to find model path
            # For now assume standard
            # In production, this should use settings.internvl_model to find path
            model_dir = self.models_dir / "ocr" / settings.internvl_model.replace("/", "-").lower()
            if not model_dir.exists():
                 # Fallback or raise
                 pass
            device = "cuda" if torch and torch.cuda.is_available() else "cpu"
            self._model = InternVLAdapter(InternVLConfig(model_dir=model_dir, device=device))
        except Exception as e:
            raise ModelLoadError(f"Failed to load InternVL: {e}")

    def unload(self) -> None:
        if self._model and hasattr(self._model, 'close'):
            self._model.close()
        self._model = None
        if torch: torch.cuda.empty_cache()

    def process_page(self, image_path: Path) -> Optional[Tuple[str, Dict[str, object]]]:
        if not self._model: self.load()
        try:
            text = self._model.predict(image_path)
            return text, {}
        except Exception as e:
            LOGGER.warning(f"InternVL failed: {e}")
            return None

class QwenVLBackend(OCRBackend):
    name = "qwenvl"
    requires_gpu = True

    def __init__(self, models_dir: Path):
        super().__init__(models_dir)
        self._model = None

    def is_available(self) -> bool:
        return True

    def load(self) -> None:
        if self._model: return
        try:
            from latexify.models.vlm_adapters import get_vlm_adapter
            model_dir = self.models_dir / "ocr" / settings.qwenvl_model.replace("/", "-").lower()
            device = "cuda" if torch and torch.cuda.is_available() else "cpu"
            self._model = get_vlm_adapter(
                "qwen-vl",
                model_id=str(model_dir),
                device=device,
                load_in_8bit=False, # Optimize for 32GB VRAM -> FP16/BF16 preferred if space allows
                load_in_4bit=False 
            )
        except Exception as e:
            raise ModelLoadError(f"Failed to load QwenVL: {e}")

    def unload(self) -> None:
        if self._model and hasattr(self._model, 'close'):
            self._model.close()
        self._model = None
        if torch: torch.cuda.empty_cache()

    def process_page(self, image_path: Path) -> Optional[Tuple[str, Dict[str, object]]]:
        if not self._model: self.load()
        try:
            text = self._model.describe(image_path, prompt="Transcribe this page to LaTeX.")
            return text, {}
        except Exception as e:
            LOGGER.warning(f"QwenVL failed: {e}")
            return None

# Factory Registry
BACKEND_REGISTRY: Dict[str, Type[OCRBackend]] = {
    "nougat": NougatBackend,
    "florence2": FlorenceBackend,
    "internvl": InternVLBackend,
    "qwenvl": QwenVLBackend,
}

class OCREngine:
    """Manager that handles backend selection and fallback."""
    
    def __init__(self, mode: str = "auto"):
        self.mode = mode
        self.backends: List[OCRBackend] = []
        self.models_dir = settings.resolve_models_dir()
        self._init_backends()

    def _init_backends(self):
        if self.mode == "auto":
            candidates = ["florence2", "nougat"]
        elif self.mode == "ensemble":
            # Best quality mix for 5090
            candidates = ["florence2", "qwenvl", "nougat"]
        elif self.mode in BACKEND_REGISTRY:
            candidates = [self.mode]
        else:
            candidates = []

        for name in candidates:
            cls = BACKEND_REGISTRY.get(name)
            if cls:
                backend = cls(self.models_dir)
                # We skip availability check here to allow lazy loading or error on load
                # But better to check if directory exists to avoid loading phantom backends
                if backend.is_available():
                    self.backends.append(backend)
        
        if not self.backends:
            LOGGER.warning("No available OCR backends found.")

    def process_page(self, page_index: int, image_path: Path) -> OCRResult:
        sources = OrderedDict()
        metadata = {}
        
        for backend in self.backends:
            try:
                result = backend.process_page(image_path)
                if result:
                    text, meta = result
                    sources[backend.name] = text
                    metadata[backend.name] = meta
                
                # Aggressive memory management for 32GB RAM / 32GB VRAM
                # If running multiple heavy models (ensemble), unload immediately
                if settings.ocr_release_mode == "page":
                    backend.unload()
                    
            except Exception as e:
                LOGGER.error(f"Backend {backend.name} failed on page {page_index}: {e}")
                
        return OCRResult(sources, metadata)

    def shutdown(self):
        for b in self.backends:
            b.unload()