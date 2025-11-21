from fastapi import APIRouter, BackgroundTasks
from sse_starlette.sse import EventSourceResponse
import asyncio
import json
from typing import AsyncGenerator

router = APIRouter()

# Mock global state for streaming
STREAM_CHANNELS = {}

@router.get("/stream/{doc_id}")
async def stream_compilation(doc_id: str):
    """
    SSE Endpoint for Streaming Partial Rendering.
    Streams 'preview_available' events when a chunk compiles successfully.
    """
    async def event_generator() -> AsyncGenerator[dict, None]:
        while True:
            # Check global state for updates on doc_id
            # In production, subscribe to Redis PubSub
            if doc_id in STREAM_CHANNELS:
                event = STREAM_CHANNELS[doc_id].pop(0) if STREAM_CHANNELS[doc_id] else None
                if event:
                    yield {
                        "event": event["type"],
                        "data": json.dumps(event["payload"])
                    }
            
            # Keep alive
            yield {"event": "ping", "data": ""}
            await asyncio.sleep(1)

    return EventSourceResponse(event_generator())

def notify_preview_ready(doc_id: str, chunk_id: str, image_url: str):
    """
    Called by the backend when a chunk is successfully compiled to an image.
    """
    if doc_id not in STREAM_CHANNELS:
        STREAM_CHANNELS[doc_id] = []
    
    STREAM_CHANNELS[doc_id].append({
        "type": "preview_ready",
        "payload": {
            "chunk_id": chunk_id,
            "image_url": image_url
        }
    })