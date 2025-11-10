from __future__ import annotations

import re
from typing import Any, Dict, List

from .base import BaseCouncilBackend, CouncilOutput, LayoutChunk


class MinerUBackend(BaseCouncilBackend):
    name = "mineru"

    async def process(self, chunk: LayoutChunk) -> CouncilOutput:
        markdown = self._to_table(chunk.text)
        metadata = {"columns": markdown["columns"], "rows": len(markdown["rows"])}
        table_md = self._render_table(markdown)
        row_count = len(markdown["rows"])
        return CouncilOutput(
            backend=self.name,
            chunk_id=chunk.chunk_id,
            page_index=chunk.page_index,
            text=table_md,
            confidence=0.6 if row_count > 1 else 0.3,
            metadata=metadata,
        )

    def _to_table(self, text: str) -> Dict[str, Any]:
        rows: List[List[str]] = []
        for line in text.splitlines():
            if not line.strip():
                continue
            cells = re.split(r"\s{3,}|\t|\s\|\s", line.strip())
            if len(cells) >= 2:
                rows.append(cells)
        if not rows:
            return {"columns": [], "rows": []}
        width = max(len(r) for r in rows)
        normalized = [r + [""] * (width - len(r)) for r in rows]
        header = normalized[0]
        body = normalized[1:] or [[""] * width]
        return {"columns": header, "rows": body}

    def _render_table(self, table: Dict[str, Any]) -> str:
        columns: List[str] = table["columns"] or ["Column 1", "Column 2"]
        header = "| " + " | ".join(columns) + " |"
        divider = "| " + " | ".join("---" for _ in columns) + " |"
        lines = [header, divider]
        for row in table["rows"]:
            lines.append("| " + " | ".join(row[: len(columns)]) + " |")
        if len(lines) == 2:
            lines.append("| " + " | ".join("" for _ in columns) + " |")
        return "\n".join(lines)


__all__ = ["MinerUBackend"]
