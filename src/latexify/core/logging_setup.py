import logging
import sys
import structlog
from typing import Optional

def configure_logging(log_level: str = "INFO", json_logs: bool = True):
    """
    Configure structured logging for the application.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    if json_logs:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Redirect standard logging to structlog
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
    )
    
    # Capture warnings
    logging.captureWarnings(True)

def get_logger(name: Optional[str] = None):
    return structlog.get_logger(name)
