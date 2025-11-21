"""
Model Warmup Utilities for RTX 5090.

Warmup is critical for:
1. torch.compile() JIT compilation 
2. CUDA kernel auto-tuning
3. Stable benchmark measurements
"""

import logging
import torch
from typing import Tuple, Optional
import time

logger = logging.getLogger(__name__)


def warmup_model(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...],
    n_iters: int = 10,
    device: str = "cuda",
    verbose: bool = True
) -> float:
    """
    Warm up model with dummy inputs to trigger JIT compilation and kernel tuning.
    
    Args:
        model: PyTorch model to warm up
        input_shape: Input tensor shape
        n_iters: Number of warmup iterations (default: 10)
        device: Target device (default: cuda)
        verbose: Log progress (default: True)
    
    Returns:
        Average warmup iteration time in milliseconds
    
    Example:
        >>> model = torch.compile(my_model)
        >>> warmup_model(model, (1, 3, 640, 640), n_iters=10)
        First iteration: 1240.2ms (JIT compilation)
        Average: 12.4ms
        12.4
    """
    model = model.to(device).eval()
    dummy = torch.randn(input_shape, device=device)
    
    if verbose:
        logger.info(f"Warming up {model.__class__.__name__} for {n_iters} iterations")
    
    times = []
    
    with torch.no_grad():
        for i in range(n_iters):
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            _ = model(dummy)
            
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000  # ms
            times.append(elapsed)
            
            if verbose and i == 0:
                logger.info(f"First iteration: {elapsed:.1f}ms (includes JIT compilation)")
    
    avg_time = sum(times[1:]) / (len(times) - 1) if len(times) > 1 else times[0]
    
    if verbose:
        logger.info(f"Warmup complete. Average: {avg_time:.1f}ms")
    
    return avg_time


def estimate_compilation_overhead(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...],
    device: str = "cuda"
) -> Tuple[float, float]:
    """
    Estimate torch.compile() overhead.
    
    Args:
        model: Model to benchmark (should be torch.compile()-ed)
        input_shape: Input shape
        device: Target device
    
    Returns:
        Tuple of (first_run_ms, steady_state_ms)
    
    Example:
        >>> model = torch.compile(my_model, mode="max-autotune")
        >>> compile_time, runtime = estimate_compilation_overhead(model, (1, 3, 640, 640))
        >>> speedup = compile_time / runtime
        >>> print(f"Compilation overhead amortized after {int(speedup)} iterations")
    """
    times = []
    model = model.to(device).eval()
    dummy = torch.randn(input_shape, device=device)
    
    with torch.no_grad():
        # First run (compile)
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = model(dummy)
        torch.cuda.synchronize()
        first_run = (time.perf_counter() - start) * 1000
        
        # Steady state (5 runs average)
        for _ in range(5):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(dummy)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
    
    steady_state = sum(times) / len(times)
    
    logger.info(
        f"Compilation overhead: {first_run:.1f}ms first run, "
        f"{steady_state:.1f}ms steady state ({first_run/steady_state:.1f}x overhead)"
    )
    
    return first_run, steady_state


def clear_cuda_cache():
    """
    Clear CUDA cache before benchmarking to ensure consistent memory state.
    
    Example:
        >>> clear_cuda_cache()
        >>> # Now run benchmark with clean slate
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        logger.debug("CUDA cache cleared")
