"""
Structured logging configuration for LaTeXify using structlog.

Provides JSON-formatted logs with contextual information for better debugging
and observability in production.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

# Try importing structlog
try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    structlog = None
    STRUCTLOG_AVAILABLE = False


def configure_logging(
    log_level: str = "INFO",
    log_format: str = "json",
    log_dir: Optional[Path] = None,
    enable_gpu_monitoring: bool = False
):
    """
    Configure structured logging for LaTeXify.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_format: Format ("json" or "text")
        log_dir: Directory for log files (if None, logs to stdout only)
        enable_gpu_monitoring: Enable GPU stats in logs
    """
    if not STRUCTLOG_AVAILABLE:
        # Fallback to standard logging
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logging.warning("structlog not available. Using standard logging.")
        return
    
    # Create log directory if specified
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(exist_ok=True, parents=True)
    
    # Configure processors
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]
    
    # Add GPU context processor if enabled
    if enable_gpu_monitoring:
        processors.append(add_gpu_context)
    
    # Output format
    if log_format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )
    
    # Add file handler if log_dir specified
    if log_dir:
        file_handler = logging.FileHandler(log_dir / "latexify.log")
        file_handler.setLevel(getattr(logging, log_level.upper()))
        logging.root.addHandler(file_handler)
    
    logging.info("Structured logging configured", extra={
        "log_level": log_level,
        "log_format": log_format,
        "log_dir": str(log_dir) if log_dir else None
    })


def add_gpu_context(logger, method_name, event_dict):
    """
    Processor to add GPU stats to log events.
    
    This is called for every log event, so keep it lightweight.
    """
    try:
        from latexify.utils.gpu_monitor import GPUMonitor
        
        # Get quick GPU stats (avoid creating new monitor each time)
        if not hasattr(add_gpu_context, 'monitor'):
            add_gpu_context.monitor = GPUMonitor()
        
        stats = add_gpu_context.monitor.get_current_stats()
        
        event_dict['gpu_vram_used_gb'] = round(stats['vram_used_gb'], 2)
        event_dict['gpu_temp_c'] = stats['temperature_c']
        event_dict['gpu_util_percent'] = stats['gpu_utilization_percent']
        
    except Exception:
        # Silently skip if GPU monitoring fails
        pass
    
    return event_dict


def get_logger(name: str):
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        structlog logger instance (or standard logger if structlog unavailable)
    """
    if STRUCTLOG_AVAILABLE:
        return structlog.get_logger(name)
    else:
        return logging.getLogger(name)


# Example usage for migration:
# OLD: logger = logging.getLogger(__name__)
# NEW: from latexify.utils.logging_config import get_logger
#      logger = get_logger(__name__)
#
# Then use structured logging:
# logger.info("bbox_detected", page=i, bbox=bbox, category=cat, confidence=conf)
