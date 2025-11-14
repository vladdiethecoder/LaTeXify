from datetime import datetime, timezone
import json
from typing import AsyncIterator, Dict, Any

from starlette.responses import EventSourceResponse


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def wrap_event(payload: Dict[str, Any]) -> Dict[str, Any]:
    if "ts" not in payload:
        payload["ts"] = now_iso()
    return payload


def sse_response(stream: AsyncIterator[Dict[str, Any]]) -> EventSourceResponse:
    async def event_generator():
        async for ev in stream:
            yield {"event": "message", "data": json.dumps(wrap_event(ev))}

    return EventSourceResponse(event_generator())
