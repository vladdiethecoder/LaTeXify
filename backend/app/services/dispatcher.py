import asyncio
from typing import AsyncIterator, Dict, Any

from .planner import plan_event
from .status_bus import status_bus, StageForwarder
from .release_bridge import build_demo_plan, plan_jobs, PIPELINE_STAGES


async def _tokenize(text: str, delay: float = 0.02):
    for ch in text:
        await asyncio.sleep(delay)
        yield ch


async def stream_build() -> AsyncIterator[Dict[str, Any]]:
    demo_plan = build_demo_plan()
    stage_forwarder = StageForwarder(status_bus)
    yield plan_event(demo_plan)

    for stage_key, description in PIPELINE_STAGES:
        await stage_forwarder.emit(stage_key, "WORKING", notes=description)
        yield {"type": "status", "status": {"stage": stage_key, "state": "WORKING", "notes": description}}
        await asyncio.sleep(0)
        await stage_forwarder.emit(stage_key, "DONE", notes=description)
        yield {"type": "status", "status": {"stage": stage_key, "state": "DONE", "notes": description}}

    for job in plan_jobs(demo_plan):
        status_payload = {"agent": job["agent"], "task": f"Generating {job['id']}", "state": "WORKING"}
        status_event = {"type": "status", "status": status_payload}
        await status_bus.publish(status_event)
        yield status_event
        async for tok in _tokenize(job["content"]):
            yield {"type": "token", "id": job["id"], "token": tok}
        yield {"type": "block_done", "id": job["id"]}
        done_payload = {"agent": job["agent"], "task": f"Finished {job['id']}", "state": "DONE"}
        done_event = {"type": "status", "status": done_payload}
        await status_bus.publish(done_event)
        yield done_event

    yield {"type": "done"}


__all__ = ["stream_build"]
