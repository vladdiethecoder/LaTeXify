from typing import AsyncIterator, Dict, Any, Optional
import logging

from ..services.dispatcher import stream_build
from ..services.release_bridge import PIPELINE_STAGES

try:
    from latexify.agents.orchestrator_graph import build_graph_manifest
except Exception:  # runtime safe guard; graph manifest is optional
    build_graph_manifest = None

LOGGER = logging.getLogger(__name__)


class GraphRunner:
    """Stateful adapter that wraps the streaming demo pipeline.

    The runner mirrors events from ``stream_build`` while tracking a coarse
    state machine (IDLE -> RUNNING -> COMPLETED / FAILED) and progress across
    the high-level pipeline stages defined in ``PIPELINE_STAGES``.
    """

    def __init__(self) -> None:
        self.enabled: bool = True
        self._state: str = "IDLE"
        self._progress: float = 0.0
        self._current_stage: Optional[str] = None

    @property
    def state(self) -> str:
        return self._state

    @property
    def progress(self) -> float:
        return self._progress

    @property
    def current_stage(self) -> Optional[str]:
        return self._current_stage

    def manifest(self) -> Dict[str, Any] | None:
        """Expose the compile–repair manifest for the streaming UI/debug views."""
        if build_graph_manifest is None:
            # Fallback manifest when the orchestrator graph is not available.
            nodes = [stage for stage, _ in PIPELINE_STAGES]
            return {
                "nodes": nodes,
                "edges": [
                    (nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)
                ],
            }
        try:
            return build_graph_manifest()
        except Exception as exc:  # pragma: no cover - runtime guard
            LOGGER.warning("Failed to build graph manifest: %s", exc)
            return None

    async def stream(self, job_id: str | None = None) -> AsyncIterator[Dict[str, Any]]:
        """Proxy events from ``stream_build`` while updating internal state."""
        if not self.enabled:
            # If disabled, delegate directly to the demo stream.
            async for event in stream_build():
                yield event
            return

        self._state = "RUNNING"
        self._progress = 0.0
        stage_keys = [key for key, _ in PIPELINE_STAGES]
        stage_index = {key: idx for idx, key in enumerate(stage_keys)}

        try:
            async for event in stream_build():
                # Update local state machine based on status/done events.
                evt_type = event.get("type")
                if evt_type == "status":
                    status = event.get("status") or {}
                    stage = status.get("stage") or status.get("agent")
                    if isinstance(stage, str) and stage in stage_index:
                        self._current_stage = stage
                        idx = stage_index[stage]
                        self._progress = 100.0 * (idx + 1) / max(len(stage_keys), 1)
                elif evt_type == "done":
                    self._state = "COMPLETED"
                    self._progress = 100.0

                # Mirror original event to the SSE client.
                yield event

            if self._state != "COMPLETED":
                self._state = "COMPLETED"
                self._progress = 100.0
        except Exception as exc:  # pragma: no cover - runtime guard
            LOGGER.error("GraphRunner stream failed: %s", exc, exc_info=True)
            self._state = "FAILED"
            error_event = {"type": "error", "error": str(exc)}
            yield error_event


graph_runner = GraphRunner()

__all__ = ["graph_runner", "GraphRunner"]
