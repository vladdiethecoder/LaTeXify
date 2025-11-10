from __future__ import annotations

import json
import logging
import sys
from typing import Any, Mapping


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter(LOG_FORMAT))

logger = logging.getLogger("latexify")
if not logger.handlers:
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)
logger.propagate = False


def configure_logging(*, verbose: bool = False, level: int | None = None) -> logging.Logger:
    """Configure the shared LaTeXify logger."""
    resolved_level = level if level is not None else (logging.DEBUG if verbose else logging.INFO)
    logger.setLevel(resolved_level)
    return logger


def _serialize(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except TypeError:
        return str(value)


def _structured_message(message: str, extra: Mapping[str, Any] | None = None) -> str:
    if not extra:
        return message
    payload = " ".join(f"{key}={_serialize(value)}" for key, value in extra.items())
    return f"{message} | {payload}"


def log_debug(message: str, **extra: Any) -> None:
    logger.debug(_structured_message(message, extra))


def log_info(message: str, **extra: Any) -> None:
    logger.info(_structured_message(message, extra))


def log_warning(message: str, **extra: Any) -> None:
    logger.warning(_structured_message(message, extra))


def log_error(message: str, **extra: Any) -> None:
    logger.error(_structured_message(message, extra))


def log_exception(message: str, **extra: Any) -> None:
    logger.exception(_structured_message(message, extra))


__all__ = [
    "configure_logging",
    "log_debug",
    "log_error",
    "log_exception",
    "log_info",
    "log_warning",
    "logger",
]
