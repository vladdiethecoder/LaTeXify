# dev/chunking/page_aware_chunker.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Optional: try to import transformers pipeline for layout; fall back if unavailable
try:
    from transformers import pipeline  # type: ignore
except Exception:  # pragma: no cover
    pipeline = None  # sentinel

PREFERRED_MODELS = [
    "nanonets-ocr2-3b",
    "nanonets-ocr-s",
    "qwen2-vl-ocr-2b-instruct",
]

def _load_layout_pipeline():
    """
    Try to load the HURIDOCS layout model; return None on any failure.
    We keep this extremely forgiving so chunking never blocks.
    """
    if pipeline is None:
        return None
    model_id = "HURIDOCS/pdf-document-layout-analysis"
    try:
        # token-classification enables simple span grouping, but many HF
        # community models lack a compatible config; catch & fall back.
        nlp = pipeline(
            task="token-classification",
            model=model_id,
            aggregation_strategy="simple",
        )
        return nlp
    except Exception as e:  # pragma: no cover
        print(f"[chunker] Layout model unavailable ({e}); falling back to heuristics.")
        return None

def _find_best_text_for_page(run_dir: Path, page_png: Path) -> Tuple[str, str]:
    """
    Choose the 'best' OCR text for this page by model preference order.
    Returns (model_name, text_md). Empty text if none found.
    """
    for model in PREFERRED_MODELS:
        md_path = run_dir / "outputs" / model / (page_png.name.replace(".png", ".md"))
        if md_path.exists():
            text = md_path.read_text(encoding="utf-8").strip()
            if text:
                return model, text
    # fallback: any .md under outputs/* for this page
    for md_path in (run_dir / "outputs").rglob(page_png.name.replace(".png", ".md")):
        text = md_path.read_text(encoding="utf-8").strip()
        if text:
            parent = md_path.parent.name
            return parent, text
    return "", ""

def _split_paragraphs(text: str, min_par_len: int = 60) -> List[str]:
    """
    Very simple paragraph splitter: split on blank lines, collapse whitespace,
    drop very short fragments.
    """
    paras: List[str] = []
    for raw in text.replace("\r\n", "\n").split("\n\n"):
        p = "\n".join(line.strip() for line in raw.strip().splitlines()).strip()
        if len(p) >= min_par_len:
            paras.append(p)
    return paras

def _slide_chunks(paragraph: str, max_chars: int, overlap: int) -> List[Tuple[int, int, str]]:
    """
    Produce character-span chunks for a single paragraph with overlap.
    Returns list of (start, end, chunk_text) spans within the paragraph string.
    """
    n = len(paragraph)
    if n <= max_chars:
        return [(0, n, paragraph)]
    chunks: List[Tuple[int, int, str]] = []
    step = max(1, max_chars - overlap)
    i = 0
    while i < n:
        j = min(n, i + max_chars)
        chunk = paragraph[i:j]
        chunks.append((i, j, chunk))
        if j == n:
            break
        i += step
    return chunks

def _page_chunks(
    page_text: str,
    page_num: int,
    page_name: str,
    source_image: str,
    model: str,
    max_chars: int,
    overlap: int,
    min_par_len: int,
    nlp=None,
) -> List[Dict]:
    """
    If a layout pipeline is available, you could augment paragraph boundaries;
    for now, we use heuristics and treat layout as advisory in the future.
    """
    # Heuristic paragraphing now; if nlp is provided, we could later refine:
    # e.g., detect HEADINGS to avoid splitting mid-section.
    paragraphs = _split_paragraphs(page_text, min_par_len=min_par_len)

    chunks: List[Dict] = []
    running_char = 0
    for p_idx, para in enumerate(paragraphs):
        spans = _slide_chunks(para, max_chars=max_chars, overlap=overlap)
        for s_idx, (s, e, chunk) in enumerate(spans):
            chunks.append(
                {
                    "chunk_id": f"p{page_num:04d}-para{p_idx:03d}-span{s_idx:03d}",
                    "page_num": page_num,
                    "page_name": page_name,
                    "source_image": source_image,
                    "model": model,
                    "start_char_in_page": running_char + s,
                    "end_char_in_page": running_char + e,
                    "text": chunk,
                }
            )
        running_char += len(para) + 2  # account for the blank-line split
    return chunks

def build_chunks_for_run(
    run_dir: Path,
    pdf_path: Path,
    max_chars: int = 800,
    overlap: int = 120,
    min_par_len: int = 60,
) -> List[Dict]:
    """
    Build page-aware chunks from OCR outputs for a given run directory.
    Returns a list of chunk dicts; writing to JSONL is done by caller.
    """
    pages_dir = run_dir / "pages"
    if not pages_dir.exists():
        raise FileNotFoundError(f"No page images in {pages_dir}. Run OCR first.")

    page_pngs = sorted(pages_dir.glob("*.png"))
    if not page_pngs:
        raise FileNotFoundError(f"No .png pages in {pages_dir}. Run OCR first.")

    nlp = _load_layout_pipeline()
    all_chunks: List[Dict] = []
    for i, page_png in enumerate(page_pngs, start=1):
        model, text = _find_best_text_for_page(run_dir, page_png)
        if not text:
            # still emit a placeholder chunk so queries can reference the page
            all_chunks.append(
                {
                    "chunk_id": f"p{i:04d}-empty-000",
                    "page_num": i,
                    "page_name": page_png.name,
                    "source_image": str(page_png),
                    "model": model or "unknown",
                    "start_char_in_page": 0,
                    "end_char_in_page": 0,
                    "text": "",
                }
            )
            continue

        chunks = _page_chunks(
            page_text=text,
            page_num=i,
            page_name=page_png.name,
            source_image=str(page_png),
            model=model or "unknown",
            max_chars=max_chars,
            overlap=overlap,
            min_par_len=min_par_len,
            nlp=nlp,
        )
        all_chunks.extend(chunks)

    return all_chunks
