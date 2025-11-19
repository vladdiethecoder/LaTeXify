"""Shared dataclasses and helpers for the simplified release pipeline."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


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
    branch: Dict[str, Any] | None = None


@dataclass
class BoundingBox:
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass
class DocumentSpan:
    text: str
    bbox: Optional[BoundingBox] = None
    confidence: Optional[float] = None
    source_backend: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentTableCell:
    row: int
    col: int
    row_span: int = 1
    col_span: int = 1
    text: str = ""
    spans: List[DocumentSpan] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentTable:
    rows: int
    cols: int
    cells: List[DocumentTableCell] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentEquation:
    latex: Optional[str] = None
    raw_text: Optional[str] = None
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentBlock:
    block_id: str
    page_number: int
    block_type: str
    text: str
    spans: List[DocumentSpan] = field(default_factory=list)
    bbox: Optional[BoundingBox] = None
    source_backend: str = "unknown"
    images: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    table: Optional[DocumentTable] = None
    equation: Optional[DocumentEquation] = None


@dataclass
class DocumentPage:
    page_number: int
    width: Optional[float] = None
    height: Optional[float] = None
    blocks: List[DocumentBlock] = field(default_factory=list)
    source_backend: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Document:
    pages: List[DocumentPage]
    source_backend: str
    version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)


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
    payload: List[Dict[str, Any]] = []
    for snippet in snippets:
        data = asdict(snippet)
        if data.get("branch") is None:
            data.pop("branch", None)
        payload.append(data)
    save_json(payload, path)


def save_document(document: Document, path: Path) -> None:
    save_json(asdict(document), path)


def load_document(path: Path) -> Document:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _document_from_dict(payload)


def _bbox_from_dict(payload: Dict[str, Any] | None) -> Optional[BoundingBox]:
    if not payload:
        return None
    return BoundingBox(
        x0=float(payload.get("x0", payload.get("left", 0.0))),
        y0=float(payload.get("y0", payload.get("top", 0.0))),
        x1=float(payload.get("x1", payload.get("right", 0.0))),
        y1=float(payload.get("y1", payload.get("bottom", 0.0))),
    )


def _span_from_dict(payload: Dict[str, Any]) -> DocumentSpan:
    return DocumentSpan(
        text=payload.get("text", ""),
        bbox=_bbox_from_dict(payload.get("bbox")),
        confidence=payload.get("confidence"),
        source_backend=payload.get("source_backend"),
        metadata=dict(payload.get("metadata") or {}),
    )


def _table_cell_from_dict(payload: Dict[str, Any]) -> DocumentTableCell:
    return DocumentTableCell(
        row=int(payload.get("row", 0)),
        col=int(payload.get("col", 0)),
        row_span=int(payload.get("row_span", 1)),
        col_span=int(payload.get("col_span", 1)),
        text=payload.get("text", ""),
        spans=[_span_from_dict(entry) for entry in payload.get("spans", [])],
        metadata=dict(payload.get("metadata") or {}),
    )


def _table_from_dict(payload: Dict[str, Any]) -> DocumentTable:
    return DocumentTable(
        rows=int(payload.get("rows", 0)),
        cols=int(payload.get("cols", 0)),
        cells=[_table_cell_from_dict(cell) for cell in payload.get("cells", [])],
        metadata=dict(payload.get("metadata") or {}),
    )


def _equation_from_dict(payload: Dict[str, Any]) -> DocumentEquation:
    return DocumentEquation(
        latex=payload.get("latex"),
        raw_text=payload.get("raw_text"),
        confidence=payload.get("confidence"),
        metadata=dict(payload.get("metadata") or {}),
    )


def _block_from_dict(payload: Dict[str, Any]) -> DocumentBlock:
    table_payload = payload.get("table")
    equation_payload = payload.get("equation")
    return DocumentBlock(
        block_id=payload.get("block_id", ""),
        page_number=int(payload.get("page_number", payload.get("page", 0))),
        block_type=payload.get("block_type", "text"),
        text=payload.get("text", ""),
        spans=[_span_from_dict(entry) for entry in payload.get("spans", [])],
        bbox=_bbox_from_dict(payload.get("bbox")),
        source_backend=payload.get("source_backend", "unknown"),
        images=list(payload.get("images", [])),
        metadata=dict(payload.get("metadata") or {}),
        table=_table_from_dict(table_payload) if table_payload else None,
        equation=_equation_from_dict(equation_payload) if equation_payload else None,
    )


def _page_from_dict(payload: Dict[str, Any]) -> DocumentPage:
    return DocumentPage(
        page_number=int(payload.get("page_number", 0)),
        width=payload.get("width"),
        height=payload.get("height"),
        blocks=[_block_from_dict(entry) for entry in payload.get("blocks", [])],
        source_backend=payload.get("source_backend", "unknown"),
        metadata=dict(payload.get("metadata") or {}),
    )


def _document_from_dict(payload: Dict[str, Any]) -> Document:
    return Document(
        pages=[_page_from_dict(entry) for entry in payload.get("pages", [])],
        source_backend=payload.get("source_backend", "unknown"),
        version=payload.get("version", "1.0"),
        metadata=dict(payload.get("metadata") or {}),
    )
