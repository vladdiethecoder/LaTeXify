#!/usr/bin/env python3
"""Emit mock SSE events to stdout for debugging."""
import asyncio
import json
from typing import Dict, Any

from backend.app.services.dispatcher import stream_build


def fmt(event: Dict[str, Any]) -> str:
    return f"data: {json.dumps(event)}\n\n"


async def main() -> None:
    async for event in stream_build():
        print(fmt(event), end="")
        if event.get("type") == "done":
            break


if __name__ == "__main__":
    asyncio.run(main())
