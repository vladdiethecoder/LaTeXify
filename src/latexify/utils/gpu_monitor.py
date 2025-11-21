"""
GPU monitoring utilities for tracking VRAM usage and temperature.

Uses pynvml (NVIDIA Management Library) to monitor RTX 5090 performance.
"""

import logging
from typing import Dict, Optional
import time
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Try importing pynvml
try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    pynvml = None
    NVML_AVAILABLE = False


class GPUMonitor:
    """
    GPU monitoring for VRAM usage, temperature, and utilization.
    
    Usage:
        monitor = GPUMonitor()
        with monitor.track("model_loading"):
            # ... load model ...
        
        stats = monitor.get_stats()
        print(f"Peak VRAM: {stats['peak_vram_gb']:.2f} GB")
    """
    
    def __init__(self, device_index: int = 0):
        self.device_index = device_index
        self.handle = None
        self.initialized = False
        self.stats_history = []
        
        if NVML_AVAILABLE:
            self._initialize()
        else:
            logger.warning("pynvml not available. GPU monitoring disabled.")
    
    def _initialize(self):
        """Initialize NVML."""
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            self.initialized = True
            
            # Log GPU info
            name = pynvml.nvmlDeviceGetName(self.handle)
            logger.info(f"GPU Monitor initialized: {name} (Device {self.device_index})")
            
        except Exception as e:
            logger.error(f"NVML initialization failed: {e}")
            self.initialized = False
    
    def get_current_stats(self) -> Dict:
        """
        Get current GPU statistics.
        
        Returns:
            Dict with keys: vram_used_gb, vram_total_gb, vram_percent,
                           temperature_c, gpu_utilization_percent
        """
        if not self.initialized:
            return self._mock_stats()
        
        try:
            # Memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            vram_used_gb = mem_info.used / (1024 ** 3)
            vram_total_gb = mem_info.total / (1024 ** 3)
            vram_percent = (mem_info.used / mem_info.total) * 100
            
            # Temperature
            temperature_c = pynvml.nvmlDeviceGetTemperature(
                self.handle,
                pynvml.NVML_TEMPERATURE_GPU
            )
            
            # GPU utilization
            utilization = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            gpu_percent = utilization.gpu
            
            return {
                "vram_used_gb": vram_used_gb,
                "vram_total_gb": vram_total_gb,
                "vram_percent": vram_percent,
                "temperature_c": temperature_c,
                "gpu_utilization_percent": gpu_percent,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.warning(f"Failed to get GPU stats: {e}")
            return self._mock_stats()
    
    def _mock_stats(self) -> Dict:
        """Return mock stats when NVML unavailable."""
        return {
            "vram_used_gb": 0.0,
            "vram_total_gb": 32.0,
            "vram_percent": 0.0,
            "temperature_c": 0,
            "gpu_utilization_percent": 0,
            "timestamp": time.time()
        }
    
    @contextmanager
    def track(self, operation_name: str):
        """
        Context manager to track GPU stats during an operation.
        
        Args:
            operation_name: Name of the operation being tracked
        """
        logger.info(f"GPU tracking started: {operation_name}")
        
        # Record start stats
        start_stats = self.get_current_stats()
        start_time = time.time()
        
        try:
            yield
        finally:
            # Record end stats
            end_stats = self.get_current_stats()
            end_time = time.time()
            duration = end_time - start_time
            
            # Compute peak VRAM
            peak_vram = max(start_stats["vram_used_gb"], end_stats["vram_used_gb"])
            vram_delta = end_stats["vram_used_gb"] - start_stats["vram_used_gb"]
            
            # Log summary
            logger.info(
                f"GPU tracking complete: {operation_name} | "
                f"Duration: {duration:.2f}s | "
                f"Peak VRAM: {peak_vram:.2f}GB | "
                f"VRAM delta: {vram_delta:+.2f}GB | "
                f"Temp: {end_stats['temperature_c']}°C"
            )
            
            # Store in history
            self.stats_history.append({
                "operation": operation_name,
                "duration": duration,
                "peak_vram_gb": peak_vram,
                "vram_delta_gb": vram_delta,
                "start_stats": start_stats,
                "end_stats": end_stats
            })
    
    def get_stats_summary(self) -> Dict:
        """
        Get summary of all tracked operations.
        
        Returns:
            Dict with peak VRAM, total duration, etc.
        """
        if not self.stats_history:
            return {"message": "No operations tracked yet"}
        
        total_duration = sum(s["duration"] for s in self.stats_history)
        peak_vram = max(s["peak_vram_gb"] for s in self.stats_history)
        avg_vram = sum(s["peak_vram_gb"] for s in self.stats_history) / len(self.stats_history)
        
        return {
            "num_operations": len(self.stats_history),
            "total_duration": total_duration,
            "peak_vram_gb": peak_vram,
            "avg_vram_gb": avg_vram,
            "operations": [
                {
                    "name": s["operation"],
                    "duration": s["duration"],
                    "peak_vram_gb": s["peak_vram_gb"]
                }
                for s in self.stats_history
            ]
        }
    
    def reset(self):
        """Reset stats history."""
        self.stats_history = []
    
    def shutdown(self):
        """Shutdown NVML."""
        if self.initialized:
            try:
                pynvml.nvmlShutdown()
                logger.info("NVML shutdown complete")
            except Exception as e:
                logger.warning(f"NVML shutdown failed: {e}")
