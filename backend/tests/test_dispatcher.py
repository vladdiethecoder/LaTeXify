import asyncio

import pytest

from backend.app.services import dispatcher


def test_stream_build_emits_plan_event():
    async def _consume():
        stream = dispatcher.stream_build()
        event = await stream.__anext__()
        assert event["type"] == "plan"
        for _ in range(3):
            await stream.__anext__()

    asyncio.run(_consume())
