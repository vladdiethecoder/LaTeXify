"""Shared dataclasses and helpers for the simplified release pipeline."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List


@dataclass
class Chunk:
    chunk_id: str
    page: int
    text: str
    images: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PlanBlock:
    block_id: str
    chunk_id: str
    label: str
    block_type: str
    images: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Snippet:
    chunk_id: str
    latex: str
    notes: Dict[str, Any] = field(default_factory=dict)


def save_jsonl(items: Iterable[Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def save_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_chunks(path: Path) -> List[Chunk]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Chunk(**item) for item in data]


def save_chunks(chunks: Iterable[Chunk], path: Path) -> None:
    save_json([asdict(chunk) for chunk in chunks], path)


def load_plan(path: Path) -> List[PlanBlock]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [PlanBlock(**item) for item in data]


def save_plan(blocks: Iterable[PlanBlock], path: Path) -> None:
    save_json([asdict(block) for block in blocks], path)


def load_snippets(path: Path) -> List[Snippet]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Snippet(**item) for item in data]


def save_snippets(snippets: Iterable[Snippet], path: Path) -> None:
    save_json([asdict(snippet) for snippet in snippets], path)
