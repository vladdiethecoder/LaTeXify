import os
from typing import Any
from pathlib import Path
from .base import MathRecognizer
import torch

# Compat: newer transformers removed apply_chunking_to_forward, but unimernet expects it.
try:
    import transformers
    from transformers import modeling_utils as _tmu

    if not hasattr(_tmu, "apply_chunking_to_forward"):
        def apply_chunking_to_forward(forward_fn, chunk_size, chunk_dim, *input_tensors):
            return forward_fn(*input_tensors)
        _tmu.apply_chunking_to_forward = apply_chunking_to_forward  # type: ignore[attr-defined]
    if not hasattr(_tmu, "find_pruneable_heads_and_indices"):
        def find_pruneable_heads_and_indices(heads, num_attention_heads, head_size, already_pruned_heads):
            return set(), None, 0
        _tmu.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices  # type: ignore[attr-defined]
    if not hasattr(_tmu, "prune_linear_layer"):
        def prune_linear_layer(layer, index, dim: int = 0):
            return layer
        _tmu.prune_linear_layer = prune_linear_layer  # type: ignore[attr-defined]
except Exception:
    transformers = None  # type: ignore

try:
    from unimernet.models.unimernet.unimernet import UniMERModel
    from unimernet.processors.formula_processor import FormulaImageEvalProcessor
    UNIMER_AVAILABLE = True
except ImportError as e:
    print(f"DEBUG: UniMERNet import failed: {e}")
    UNIMER_AVAILABLE = False

class UniMERNetMathRecognizer(MathRecognizer):
    def __init__(self, cfg_path: str = "config/model/unimer.yaml", device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.mock_mode = not UNIMER_AVAILABLE
        
        if not UNIMER_AVAILABLE:
            print("WARNING: UniMERNet not installed. Running in MOCK mode for Math Recognition.")
            self.model = None
            self.processor = None
        else:
            try:
                # Try to locate models
                from latexify.core.model_paths import resolve_models_root
                # Assuming we are in repo root or src/latexify is importable
                # We can guess the path relative to cwd if resolve_models_root fails or is not configured
                repo_root = Path(".").resolve()
                models_dir = repo_root / "models"
                if not models_dir.exists():
                     # Fallback for release/run_release.py context
                     models_dir = repo_root / "release" / "models"
                
                model_path = models_dir / "ocr" / "unimernet"
                
                from unimernet.models.unimernet.unimernet import UniMERModel
                from omegaconf import OmegaConf

                if not model_path.exists():
                    print(f"WARNING: UniMERNet model not found at {model_path}. Using mock.")
                    self.mock_mode = True
                    return

                print(f"Loading UniMERNet from local path: {model_path}")
                checkpoint_path = model_path / "pytorch_model.pth"
                
                # 1. Load default config
                # We can't easily get the internal yaml path without using the class method, 
                # but UniMERModel inherits from BaseModel which has default_config_path.
                # Let's assume standard registry usage.
                from unimernet.common.registry import registry
                # Ensure model is registered
                # (Importing UniMERModel should have registered it)
                
                config_path = UniMERModel.default_config_path("unimernet")
                cfg = OmegaConf.load(config_path)
                
                # 2. Patch config to point to OUR local model path
                cfg.model.tokenizer_config.path = str(model_path)
                cfg.model.model_name = str(model_path)
                
                # Inject model_name into model_config because UniMERModel expects it there
                if "model_name" not in cfg.model.model_config:
                    cfg.model.model_config.model_name = cfg.model.model_name
                else:
                    cfg.model.model_config.model_name = str(model_path)

                # 3. Instantiate model with explicit arguments
                model_args = {
                    "model_name": cfg.model.model_name,
                    "model_config": cfg.model.model_config,
                    "tokenizer_name": cfg.model.tokenizer_name,
                    "tokenizer_config": cfg.model.tokenizer_config,
                }
                self.model = UniMERModel(**model_args).to(self.device)
                
                # 4. Load checkpoint
                self.model.load_checkpoint(str(checkpoint_path))
                self.model.eval()
                
                # 5. Load processor
                from unimernet.processors.formula_processor import FormulaImageEvalProcessor
                self.processor = FormulaImageEvalProcessor(image_size=(192, 672))
                
                print("UniMERNet loaded successfully.")
            except Exception as e:
                print(f"ERROR: Failed to load UniMERNet: {e}")
                import traceback
                traceback.print_exc()
                self.mock_mode = True

    def predict(self, image: Any) -> str:
        if self.mock_mode:
            return "\\text{Mock Equation: } E = mc^2"
            
        try:
            # Processor expects image (PIL or np)
            pixel_values = self.processor(image).unsqueeze(0).to(self.device)
            
            # Generate
            output = self.model.generate({"image": pixel_values})
            if isinstance(output, list):
                return output[0]
            return output
        except Exception as e:
            print(f"UniMERNet prediction failed: {e}")
            return "\\text{Error}"

    def predict_batch(self, images: list[Any]) -> list[str]:
        if self.mock_mode:
            return [f"\\text{{Mock Batch Eq {i}}}" for i in range(len(images))]
        
        return [self.predict(img) for img in images]
