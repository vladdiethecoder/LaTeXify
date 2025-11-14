import asyncio
from typing import AsyncIterator, Dict, Any, List

from .planner import plan_event
from .status_bus import status_bus


async def _tokenize(text: str, delay: float = 0.02):
    for ch in text:
        await asyncio.sleep(delay)
        yield ch


async def stream_build() -> AsyncIterator[Dict[str, Any]]:
    yield plan_event()

    agents: List[Dict[str, Any]] = [
        {
            "agent": "TextAgent",
            "id": "c1_text_1",
            "content": "This is a streamed paragraph for column 1.",
        },
        {
            "agent": "CodeAgent",
            "id": "c1_code_1",
            "content": "\\begin{mdframed}\\textbf{Example}\\end{mdframed}",
        },
        {
            "agent": "EquationAgent",
            "id": "c2_math_1",
            "content": "\\sigma_n = \\begin{pmatrix} \\cos\\theta & -\\sin\\theta \\\\ \\sin\\theta & \\cos\\theta \\end{pmatrix}",
        },
        {
            "agent": "TextAgent",
            "id": "c2_text_2",
            "content": "Additional notes rendered live on the right.",
        },
    ]

    for job in agents:
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
