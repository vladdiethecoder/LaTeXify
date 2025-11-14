from typing import AsyncIterator, Dict, Any


class GraphRunner:
    """Skeleton adapter for future LangGraph integration."""

    def __init__(self) -> None:
        self.enabled = False

    async def stream(self) -> AsyncIterator[Dict[str, Any]]:
        raise NotImplementedError("LangGraph runner not wired yet.")


graph_runner = GraphRunner()

__all__ = ["graph_runner", "GraphRunner"]
