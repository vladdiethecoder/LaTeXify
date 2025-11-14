"""Placeholder equation node for LangGraph streaming."""


class EquationNode:
    def __init__(self) -> None:
        self.name = "EquationNode"

    async def run(self, *args, **kwargs):
        raise NotImplementedError("LangGraph equation node not implemented.")
