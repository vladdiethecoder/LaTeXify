"""
Optimization utilities for RTX 5090 hardware acceleration.

Modules:
- fp8_utils: FP8 quantization using torchao
- cuda_graphs: CUDA Graph wrappers for kernel fusion
- warmup: Model warmup utilities
"""

from .fp8_utils import apply_fp8_quantization
from .cuda_graphs import CUDAGraphWrapper
from .warmup import warmup_model

__all__ = [
    "apply_fp8_quantization",
    "CUDAGraphWrapper",
    "warmup_model",
]
