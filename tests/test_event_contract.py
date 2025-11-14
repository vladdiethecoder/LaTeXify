import asyncio
import json
from pathlib import Path

import fastjsonschema
import pytest

from backend.app.services import dispatcher

EVENT_SCHEMA_PATH = Path("schemas/events.schema.json")


async def collect_events():
    events = []
    async for event in dispatcher.stream_build():
        events.append(event)
        if event.get("type") == "done":
            break
    return events


@pytest.mark.asyncio
async def test_event_stream_matches_schema():
    schema = json.loads(EVENT_SCHEMA_PATH.read_text(encoding="utf-8"))
    validate = fastjsonschema.compile(schema)
    events = await collect_events()
    assert events[0]["type"] == "plan"
    for event in events:
        validate(event)
    assert events[-1]["type"] == "done"
