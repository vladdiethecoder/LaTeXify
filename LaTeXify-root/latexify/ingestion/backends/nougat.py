from __future__ import annotations

import re

from .base import BaseCouncilBackend, CouncilOutput, LayoutChunk


class NougatBackend(BaseCouncilBackend):
    name = "nougat"
    EQUATION_RX = re.compile(r"([A-Za-z0-9\s]+=[^\\n]+)")

    async def process(self, chunk: LayoutChunk) -> CouncilOutput:
        latex = self._extract_equations(chunk.text)
        meta = {"equation_count": latex.count("\\[")}
        return CouncilOutput(
            backend=self.name,
            chunk_id=chunk.chunk_id,
            page_index=chunk.page_index,
            text=latex or "\\[ \\text{No equations detected} \\]",
            confidence=0.55 if latex else 0.2,
            metadata=meta,
        )

    def _extract_equations(self, text: str) -> str:
        matches = self.EQUATION_RX.findall(text)
        if not matches:
            return ""
        lines = [f"\\[{m.strip()}\\]" for m in matches[:6]]
        return "\n".join(lines)


__all__ = ["NougatBackend"]
