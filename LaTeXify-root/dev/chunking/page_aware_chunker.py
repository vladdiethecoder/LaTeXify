# dev/chunking/page_aware_chunker.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

# ---------------------------
# Public entry point
# ---------------------------

def build_chunks_for_run(
    run_dir: Path,
    pdf_path: Path,
    *,
    max_chars: int = 1200,
    overlap: int = 150,
    min_par_len: int = 20,
    write_path: Path | None = None,
) -> Path:
    """
    Build page-aware chunks from OCR outputs for a given run directory.

    Inputs:
      - run_dir: dev/runs/<STAMP>
        expects:
          pages/               # page images (not required for chunking)
          outputs/<model>/*.md # OCR per-page Markdown
      - pdf_path: original source (recorded in metadata only)
      - max_chars: target max characters per chunk
      - overlap: character overlap between adjacent chunks
      - min_par_len: minimum paragraph length to keep (in characters)
      - write_path: override output jsonl path; defaults to run_dir/'chunks.jsonl'

    Output:
      - JSONL at run_dir/chunks.jsonl with fields:
          {id, page, page_idx, text, char_start, char_end,
           source_files, models, run_dir, pdf, strategy}
    """
    run_dir = Path(run_dir)
    outputs_root = run_dir / "outputs"
    if not outputs_root.exists():
        raise FileNotFoundError(f"No OCR outputs found under {outputs_root}")

    # Collect per-page, across all models
    per_page_texts: Dict[str, Dict[str, str]] = _gather_texts(outputs_root)

    # Build consensus per page (choose longest non-empty text as the base)
    consensus: Dict[str, Tuple[str, List[Path], List[str]]] = {}
    for page_name, model_map in sorted(per_page_texts.items()):
        items = [(m, t) for m, t in model_map.items() if _nonempty(t)]
        if not items:
            continue
        # choose the longest text as consensus seed
        items.sort(key=lambda kv: len(kv[1]), reverse=True)
        best_model, best_text = items[0]
        file_paths = _page_files_for_page(outputs_root, page_name)
        consensus[page_name] = (best_text, file_paths, list(model_map.keys()))

    # Chunk per page
    chunks: List[Dict] = []
    page_to_idx = _page_index_map(run_dir)
    counter = 0
    for page_name, (text, files, models) in sorted(consensus.items(), key=lambda kv: kv[0]):
        page_idx = page_to_idx.get(page_name, None)
        paragraphs = _segment_paragraphs(text)
        paragraphs = [p for p in paragraphs if len(p.strip()) >= min_par_len]

        # merge paragraphs into rolling windows with overlap
        for chunk_txt, span in _rolling_pack(paragraphs, max_chars=max_chars, overlap=overlap):
            char_start, char_end = span
            chunks.append({
                "id": f"chunk-{counter:06d}",
                "page": page_name,
                "page_idx": page_idx,
                "text": chunk_txt,
                "char_start": char_start,
                "char_end": char_end,
                "source_files": [str(p) for p in files],
                "models": models,
                "run_dir": str(run_dir),
                "pdf": str(pdf_path),
                "strategy": {
                    "type": "page_aware_longest_text",
                    "max_chars": max_chars,
                    "overlap": overlap,
                    "min_par_len": min_par_len,
                },
            })
            counter += 1

    if not chunks:
        raise RuntimeError("No chunks produced. Check OCR outputs exist and contain text.")

    out_path = write_path or (run_dir / "chunks.jsonl")
    _write_jsonl(out_path, (c for c in chunks))
    return out_path


# ---------------------------
# Helpers
# ---------------------------

def _gather_texts(outputs_root: Path) -> Dict[str, Dict[str, str]]:
    """
    Returns:
      {page_name: {model_name: text_md}}
    """
    per_page: Dict[str, Dict[str, str]] = {}
    for model_dir in sorted(outputs_root.glob("*")):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        for md_path in sorted(model_dir.glob("*.md")):
            page_name = md_path.name.replace(".md", ".png")
            txt = md_path.read_text(encoding="utf-8", errors="ignore")
            per_page.setdefault(page_name, {})[model_name] = txt
    return per_page


def _page_files_for_page(outputs_root: Path, page_name_png: str) -> List[Path]:
    """
    Return all OCR output files (.md) that correspond to page_name_png across models.
    """
    md_name = page_name_png.replace(".png", ".md")
    paths: List[Path] = []
    for model_dir in sorted(outputs_root.glob("*")):
        candidate = model_dir / md_name
        if candidate.exists():
            paths.append(candidate)
    return paths


def _page_index_map(run_dir: Path) -> Dict[str, int]:
    """
    Map page file name -> 1-based page index, using the pages/ directory.
    """
    pages_dir = run_dir / "pages"
    mapping: Dict[str, int] = {}
    if pages_dir.exists():
        pages = sorted(pages_dir.glob("*.png"))
        for i, p in enumerate(pages, start=1):
            mapping[p.name] = i
    return mapping


_PARA_SPLIT_RE = re.compile(r"\n{2,}")  # blank-line separated paragraphs
_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6}\s+|[A-Z][A-Z0-9 \-]{4,}:|[A-Z][\w \-]{4,}$)")

def _segment_paragraphs(text: str) -> List[str]:
    """
    Segment simple Markdown-ish text into paragraphs, nudging around headings.
    """
    raw_pars = [p.strip() for p in _PARA_SPLIT_RE.split(text or "") if p.strip()]
    paragraphs: List[str] = []
    buff: List[str] = []
    for p in raw_pars:
        if _HEADING_RE.match(p) and buff:
            # flush previous group at heading boundary
            paragraphs.append("\n\n".join(buff).strip())
            buff = [p]
        else:
            buff.append(p)
    if buff:
        paragraphs.append("\n\n".join(buff).strip())
    return paragraphs


def _rolling_pack(paragraphs: List[str], *, max_chars: int, overlap: int) -> Iterable[Tuple[str, Tuple[int, int]]]:
    """
    Produce character-windowed chunks with overlap from a list of paragraphs.

    Yields:
      (chunk_text, (global_char_start, global_char_end))
      where global_* are positions in the concatenated page text.
    """
    # Work in concatenated text space to record a page-relative span
    cat = ""
    spans: List[Tuple[int, int]] = []  # span per paragraph
    cursor = 0
    for p in paragraphs:
        start = cursor
        cat += (p + "\n\n")
        cursor = len(cat)
        spans.append((start, cursor))

    i = 0
    while i < len(paragraphs):
        # pack as many paragraphs as we can into max_chars
        start_span = spans[i][0]
        j = i
        while j < len(paragraphs) and (spans[j][1] - start_span) <= max_chars:
            j += 1
        # j is first paragraph that doesn't fit
        end_span = spans[j - 1][1]
        text = cat[start_span:end_span].strip()
        yield (text, (start_span, end_span))

        if j >= len(paragraphs):
            break
        # compute next window start using overlap
        window_len = end_span - start_span
        target_next_start = end_span - min(overlap, max(0, window_len // 3))
        # advance i until paragraph boundary crosses target_next_start
        k = i
        while k < len(paragraphs) and spans[k][0] < target_next_start:
            k += 1
        i = max(k, i + 1)  # ensure progress even if paragraphs are very long


def _write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _nonempty(s: str | None) -> bool:
    return bool(s and s.strip())
