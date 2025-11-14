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


status_bus = StatusBus()

__all__ = ["status_bus", "StatusBus"]
