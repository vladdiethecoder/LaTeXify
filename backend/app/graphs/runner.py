from typing import AsyncIterator, Dict, Any

try:
    from release.agents.orchestrator_graph import build_graph_manifest
except Exception:  # runtime safe guard; graph manifest is optional
    build_graph_manifest = None


class GraphRunner:
    """Skeleton adapter for future LangGraph integration."""

    def __init__(self) -> None:
        self.enabled = False

    def manifest(self) -> Dict[str, Any] | None:
        """Expose the compile–repair manifest for the streaming UI/debug views."""
        if build_graph_manifest is None:
            return None
        try:
            return build_graph_manifest()
        except Exception:
            return None

    async def stream(self) -> AsyncIterator[Dict[str, Any]]:
        raise NotImplementedError("LangGraph runner not wired yet.")


graph_runner = GraphRunner()

__all__ = ["graph_runner", "GraphRunner"]
