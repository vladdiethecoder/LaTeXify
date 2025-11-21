"""
CUDA Graphs for Kernel Fusion.

Reduces kernel launch overhead by 30-50% for fixed-size batch inference.
Critical for RTX 5090 to maximize throughput on repeated operations.

IMPORTANT: CUDA Graphs require:
1. Fixed input shapes (no dynamic batching)
2. Deterministic operations (no random sampling)
3. No CPU synchronization inside graph
"""

import logging
import torch
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class CUDAGraphWrapper:
    """
    Wraps a PyTorch module for CUDA Graph execution.
    
    Benefits:
    - Eliminates per-iteration kernel launch overhead
    - Fuses consecutive kernels (layout detection, math recognition)
    - Ideal for repeated inference with same input shape
    
    Limitations:
    - Input shape must be fixed (padding required for variable inputs)
    - Model must be deterministic (no dropout, random aug at inference)
    - Not compatible with dynamic control flow
    
    Example:
        >>> model = YOLO("doclayout-yolo.pt").model
        >>> wrapped = CUDAGraphWrapper(model, input_shape=(1, 3, 640, 640))
        >>> # First call records graph
        >>> output = wrapped(image_tensor)  
        >>> # Subsequent calls replay graph (much faster)
        >>> output = wrapped(another_image_tensor)
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        input_shape: Tuple[int, ...],
        warmup_iters: int = 3,
        device: str = "cuda"
    ):
        """
        Initialize CUDA Graph wrapper.
        
        Args:
            model: PyTorch model to wrap
            input_shape: Fixed input shape (e.g., (1, 3, 640, 640))
            warmup_iters: Number of warmup iterations before recording (default: 3)
            device: Target device (default: cuda)
        """
        self.model = model.to(device).eval()
        self.device = device
        self.input_shape = input_shape
        self.warmup_iters = warmup_iters
        
        # Static tensors for graph
        self.static_input = torch.zeros(input_shape, device=device)
        self.static_output: Optional[torch.Tensor] = None
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        
        # Record graph
        self._record_graph()
        logger.info(f"CUDA Graph recorded for {model.__class__.__name__}")
    
    def _record_graph(self):
        """Record CUDA Graph after warmup."""
        # Warmup: Compile kernels and stabilize memory allocations
        logger.debug(f"Warming up for {self.warmup_iters} iterations")
        with torch.no_grad():
            for _ in range(self.warmup_iters):
                _ = self.model(self.static_input)
        
        torch.cuda.synchronize()
        
        # Record graph
        logger.debug("Recording CUDA Graph")
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_output = self.model(self.static_input)
        
        logger.info(
            f"Graph recorded: input={self.input_shape}, "
            f"output={self.static_output.shape if hasattr(self.static_output, 'shape') else 'N/A'}"
        )
    
    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Execute model via CUDA Graph replay.
        
        Args:
            x: Input tensor (MUST match self.input_shape)
        
        Returns:
            Model output
        
        Raises:
            ValueError: If input shape mismatch
        """
        if x.shape != self.input_shape:
            raise ValueError(
                f"Input shape {x.shape} does not match graph shape {self.input_shape}. "
                "CUDA Graphs require fixed input shapes. Consider padding inputs."
            )
        
        # Copy input to static buffer
        self.static_input.copy_(x)
        
        # Replay graph (extremely fast!)
        self.graph.replay()
        
        # Return cloned output (static_output will be overwritten on next call)
        return self.static_output.clone()
    
    def reset(self):
        """Re-record graph (use if model weights changed)."""
        logger.info("Resetting CUDA Graph")
        self.graph = None
        self.static_output = None
        self._record_graph()


def supports_cuda_graphs() -> bool:
    """
    Check if current environment supports CUDA Graphs.
    
    Returns:
        True if CUDA 11.0+ with compatible GPU
    """
    if not torch.cuda.is_available():
        return False
    
    # CUDA Graphs require CUDA 11.0+
    cuda_version = torch.version.cuda
    if cuda_version is None:
        return False
    
    major, minor = map(int, cuda_version.split(".")[:2])
    if major < 11:
        logger.warning(f"CUDA Graphs require CUDA 11.0+, found {cuda_version}")
        return False
    
    return True
