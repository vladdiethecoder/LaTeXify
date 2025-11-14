from fastapi import APIRouter

from ..adapters.sse import sse_response
from ..services.dispatcher import stream_build
from ..graphs.runner import graph_runner

router = APIRouter()


@router.get("/events")
async def events():
    if graph_runner.enabled:
        return sse_response(graph_runner.stream())
    return sse_response(stream_build())
