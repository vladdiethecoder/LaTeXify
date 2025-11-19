import asyncio
from typing import AsyncIterator, Dict, Any


class StatusBus:
    """Lightweight pub/sub bus for streaming agent status events."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()

    async def publish(self, event: Dict[str, Any]) -> None:
        await self._queue.put(event)

    async def stream(self) -> AsyncIterator[Dict[str, Any]]:
        while True:
            event = await self._queue.get()
            yield event


class StageForwarder:
    """Bridges synchronous stage events into the async status bus."""

    def __init__(self, bus: StatusBus) -> None:
        self._bus = bus

    async def emit(self, stage: str, state: str, **metadata: Any) -> None:
        payload = {"stage": stage, "state": state, **metadata}
        await self._bus.publish({"type": "status", "status": payload})


status_bus = StatusBus()

__all__ = ["status_bus", "StatusBus", "StageForwarder"]
