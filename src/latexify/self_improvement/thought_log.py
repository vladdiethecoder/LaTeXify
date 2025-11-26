from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ThoughtEntry:
    iteration: int
    agent_id: str
    strategy: Optional[str]
    summary: str
    details: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)

    def to_markdown(self) -> str:
        return (
            f"## Iteration {self.iteration} | Agent {self.agent_id} | Strategy {self.strategy or 'NA'}\n"
            f"- Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.timestamp))}\n"
            f"- Summary: {self.summary}\n"
            f"- Details: {json.dumps(self.details, ensure_ascii=True)}\n\n"
        )


class ThoughtLog:
    """
    Persistent thought document with append-only semantics.
    """

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("# Self-Improvement Thought Log\n\n", encoding="utf-8")

    def append(self, entry: ThoughtEntry) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(entry.to_markdown())

    def load_recent(self, limit: int = 5) -> List[ThoughtEntry]:
        if not self.path.exists():
            return []
        content = self.path.read_text(encoding="utf-8").splitlines()
        entries: List[ThoughtEntry] = []
        for line in content:
            if line.startswith("- Details: "):
                try:
                    data = json.loads(line.replace("- Details: ", ""))
                    entries.append(
                        ThoughtEntry(
                            iteration=data.get("iteration", -1),
                            agent_id=data.get("agent_id", "unknown"),
                            strategy=data.get("strategy"),
                            summary=data.get("summary", ""),
                            details=data.get("details", {}),
                            timestamp=data.get("timestamp", time.time()),
                        )
                    )
                except json.JSONDecodeError:
                    continue
        return entries[-limit:]

    def append_json(self, raw: Dict[str, Any]) -> None:
        """
        Convenience for writing structured records.
        """
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(raw, ensure_ascii=True) + "\n")
