"""Shared telemetry helpers for Kimi-K2 inference."""
from __future__ import annotations

import threading
from typing import Dict

_LOCK = threading.Lock()
_METRICS: Dict[str, float] = {
    "calls": 0.0,
    "success": 0.0,
    "total_time": 0.0,
    "repair_attempts": 0.0,
    "repair_success": 0.0,
}


def record_inference(duration: float, success: bool) -> None:
    with _LOCK:
        _METRICS["calls"] += 1
        if success:
            _METRICS["success"] += 1
        _METRICS["total_time"] += max(0.0, duration)


def record_repair(success: bool) -> None:
    with _LOCK:
        _METRICS["repair_attempts"] += 1
        if success:
            _METRICS["repair_success"] += 1


def snapshot() -> Dict[str, float]:
    with _LOCK:
        return dict(_METRICS)


def reset() -> None:
    with _LOCK:
        for key in _METRICS:
            _METRICS[key] = 0.0


__all__ = ["record_inference", "record_repair", "snapshot", "reset"]
