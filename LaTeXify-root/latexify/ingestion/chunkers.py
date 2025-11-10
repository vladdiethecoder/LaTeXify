from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence

from .backends.base import LayoutChunk

HEADING_RX = re.compile(
    r"^(chapter|section|appendix|lesson|unit|part|module)\b", re.IGNORECASE
)
FIGURE_RX = re.compile(r"^(figure|fig\.|table)\b", re.IGNORECASE)
STRONG_HEADING_RX = re.compile(r"^[A-Z0-9 .:()-]{5,}$")


def _split_paragraphs(text: str) -> List[str]:
    parts = re.split(r"\n\s*\n", text.replace("\r\n", "\n"))
    return [p.strip() for p in parts if p.strip()]


def _looks_like_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if HEADING_RX.match(stripped):
        return True
    if FIGURE_RX.match(stripped):
        return True
    if STRONG_HEADING_RX.match(stripped) and len(stripped.split()) <= 12:
        return True
    return False


def _split_semantic_units(paragraphs: List[str]) -> List[str]:
    units: List[str] = []
    current: List[str] = []
    for para in paragraphs:
        if _looks_like_heading(para):
            if current:
                units.append("\n\n".join(current))
                current = []
            units.append(para)
        else:
            current.append(para)
    if current:
        units.append("\n\n".join(current))
    return units


def _chunk_image(images_by_page: Mapping[int, List[Path]] | None, page_index: int, chunk_idx: int) -> Path | None:
    if not images_by_page:
        return None
    candidates = images_by_page.get(page_index) or []
    if not candidates:
        return None
    if chunk_idx < len(candidates):
        return candidates[chunk_idx]
    return candidates[-1]


def semantic_chunk_pages(
    pages: Sequence[str],
    *,
    chunk_chars: int = 1100,
    min_chars: int = 320,
    images_by_page: Mapping[int, List[Path]] | None = None,
) -> List[LayoutChunk]:
    """Chunk text by semantic paragraphs with spillover handling."""

    chunks: List[LayoutChunk] = []
    for page_index, text in enumerate(pages):
        paragraphs = _split_paragraphs(text)
        paragraphs = _split_semantic_units(paragraphs)
        if not paragraphs:
            paragraphs = [text]
        buffer: List[str] = []
        counter = 0
        for para in paragraphs:
            buffer.append(para)
            joined = "\n\n".join(buffer)
            if len(joined) >= chunk_chars:
                image_path = _chunk_image(images_by_page, page_index, counter)
                chunks.append(
                    LayoutChunk(
                        chunk_id=f"page{page_index+1:04d}-chunk{counter+1:03d}",
                        page_index=page_index,
                        text=joined.strip(),
                        page_name=f"page-{page_index+1:04d}.md",
                        bbox=None,
                        image_path=image_path,
                    )
                )
                buffer = []
                counter += 1
        if buffer and (len(buffer[0]) >= min_chars or not chunks):
            image_path = _chunk_image(images_by_page, page_index, counter)
            chunks.append(
                LayoutChunk(
                    chunk_id=f"page{page_index+1:04d}-chunk{counter+1:03d}",
                    page_index=page_index,
                    text="\n\n".join(buffer).strip(),
                    page_name=f"page-{page_index+1:04d}.md",
                    image_path=image_path,
                )
            )
    return chunks


def fixed_chunk_pages(
    pages: Sequence[str],
    *,
    chunk_chars: int = 1100,
    images_by_page: Mapping[int, List[Path]] | None = None,
) -> List[LayoutChunk]:
    """Chunk pages with fixed character windows."""

    chunks: List[LayoutChunk] = []
    for page_index, text in enumerate(pages):
        cleaned = (text or "").strip()
        if not cleaned:
            continue
        start = 0
        counter = 0
        while start < len(cleaned):
            end = min(len(cleaned), start + chunk_chars)
            snippet = cleaned[start:end].strip()
            if not snippet:
                break
            image_path = _chunk_image(images_by_page, page_index, counter)
            chunks.append(
                LayoutChunk(
                    chunk_id=f"page{page_index+1:04d}-chunk{counter+1:03d}",
                    page_index=page_index,
                    text=snippet,
                    page_name=f"page-{page_index+1:04d}.md",
                    image_path=image_path,
                )
            )
            counter += 1
            start = end
    return chunks


__all__ = ["semantic_chunk_pages", "fixed_chunk_pages"]
