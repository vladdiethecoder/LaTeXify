"""Placeholder text node for LangGraph streaming."""


class TextNode:
    def __init__(self) -> None:
        self.name = "TextNode"

    async def run(self, *args, **kwargs):
        raise NotImplementedError("LangGraph text node not implemented.")
