import re
import logging
from collections import Counter, defaultdict, OrderedDict
from typing import List, Dict, Tuple, Set, Sequence, Any

from latexify.core.common import Chunk
try:
    from latexify.pipeline.semantic_chunking import SemanticChunker
except ImportError:
    SemanticChunker = None

LOGGER = logging.getLogger(__name__)

# Regex Constants
QUESTION_RE = re.compile(r"^(question|q)\s*([0-9]+[a-z]?|\([^)]+\))", re.IGNORECASE)
ANSWER_RE = re.compile(r"^(answer|solution)\b", re.IGNORECASE)
TABLE_BORDER_RE = re.compile(r"(\+[-+]+\+)|(\|.+\|)")
LIST_BULLET_RE = re.compile(r"^([0-9]+\.[\)\s]|[A-Za-z]\.|[-*•])\s+")
FORMULA_RE = re.compile(r"(\\begin\{equation\}|\\frac|\\sum|\\int|=|\[a-z]+)")
HEADER_KEYWORDS = ("chapter", "section", "appendix", "part", "lesson")
WORD_RE = re.compile(r"[A-Za-z]+")
DIGIT_GAP_RE = re.compile(r"(?<=\d)\s+(?=\d)")

def detect_header_level(line: str) -> int:
    if not line:
        return 0
    normalized = line.strip()
    lower = normalized.lower()
    if any(lower.startswith(keyword) for keyword in HEADER_KEYWORDS):
        return 1 if lower.startswith("chapter") else 2
    if normalized.isupper() and len(normalized.split()) <= 8:
        return 1
    if len(normalized.split()) <= 10 and normalized.endswith(":"):
        return 2
    return 0

def extract_table_signature(lines: List[str]) -> Dict[str, int]:
    rows = []
    max_cols = 0
    for line in lines:
        if "|" in line:
            cells = [cell for cell in line.split("|") if cell.strip()]
            if cells:
                rows.append(line)
                max_cols = max(max_cols, len(cells))
    return {"rows": len(rows), "columns": max_cols or 1}

def estimate_list_depth(lines: List[str]) -> int:
    depth = 1
    for line in lines:
        if LIST_BULLET_RE.match(line.strip()):
            indent = len(line) - len(line.lstrip(" "))
            depth = max(depth, indent // 2 + 1)
    return depth

def classify_region(text: str, has_page_images: bool, figure_hint: bool = False) -> Tuple[str, Dict[str, object]]:
    lines = [line for line in text.splitlines() if line.strip()]
    first_line = lines[0].strip() if lines else ""
    header_level = detect_header_level(first_line)
    metadata: Dict[str, object] = {"header_level": header_level, "formula_detected": False}
    trimmed = text.strip()
    if figure_hint or (has_page_images and not trimmed):
        metadata["region_type"] = "figure"
        return "figure", metadata
    if any(TABLE_BORDER_RE.search(line) for line in lines):
        metadata["table_signature"] = extract_table_signature(lines)
        metadata["region_type"] = "table"
        return "table", metadata
    list_hits = sum(1 for line in lines if LIST_BULLET_RE.match(line.strip()))
    if list_hits >= max(1, len(lines) // 2):
        metadata["list_depth"] = estimate_list_depth(lines)
        metadata["region_type"] = "list"
        return "list", metadata
    if FORMULA_RE.search(text):
        metadata["formula_detected"] = True
        metadata["region_type"] = "formula"
        return "formula", metadata
    metadata["region_type"] = "text" if header_level == 0 else "heading"
    return ("text" if header_level == 0 else "heading"), metadata

def detect_paragraph_region(paragraph: str) -> str:
    para = paragraph.strip()
    if not para:
        return "text"
    lowered = para.lower()
    if lowered.startswith("figure") or lowered.startswith("fig "):
        return "figure"
    if any(TABLE_BORDER_RE.search(line) for line in para.splitlines()):
        return "table"
    if LIST_BULLET_RE.match(para):
        return "list"
    if FORMULA_RE.search(para):
        return "formula"
    if detect_header_level(para.splitlines()[0]) > 0:
        return "heading"
    return "text"

def noise_metrics(text: str) -> Dict[str, float]:
    total_chars = len(text) or 1
    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    digits = sum(1 for ch in text if ch.isdigit())
    symbols = sum(1 for ch in text if ch in {"@", "#", "$", "%", "^", "&"})
    whitespace_runs = sum(1 for part in text.split() if len(part) == 1 and not part.isalpha())
    newline_count = text.count("\n") or 1
    avg_line_len = total_chars / newline_count
    broken_words = text.count("-\n")
    tokens = WORD_RE.findall(text)
    unique_ratio = len(set(tokens)) / len(tokens) if tokens else 0.0
    score = (
        0.4 * (non_ascii / total_chars)
        + 0.2 * (broken_words / newline_count)
        + 0.2 * (symbols / total_chars)
        + 0.2 * (1 - unique_ratio)
    )
    return {
        "non_ascii_ratio": non_ascii / total_chars,
        "digit_ratio": digits / total_chars,
        "symbol_ratio": symbols / total_chars,
        "avg_line_length": avg_line_len,
        "broken_words": broken_words,
        "unique_token_ratio": unique_ratio,
        "noise_score": min(1.0, score),
        "format_split_errors": whitespace_runs / max(1, len(tokens)),
    }

def ocr_consensus_score(transcripts: Sequence[str]) -> float:
    normalized = []
    for text in transcripts:
        candidate = " ".join(text.split()).strip().lower()
        if candidate:
            normalized.append(candidate)
    if not normalized:
        return 0.0
    counts: Counter[str] = Counter(normalized)
    return max(counts.values()) / len(normalized)

def aggressive_math_cleanup(text: str) -> str:
    replacements = {
        "−": "-",
        "–": "-",
        "—": "-",
        "∗": "*",
        "ﬁ": "fi",
        "ﬂ": "fl",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = DIGIT_GAP_RE.sub("", text)
    text = re.sub(r"([=+\-])\s{2,}", r"\1 ", text)
    text = re.sub(r"\s{2,}([=+\-])", r" \1", text)
    return text

def merge_text_sources(sources: Sequence[str]) -> str:
    merged: List[str] = []
    seen = set()
    for text in sources:
        for paragraph in (seg.strip() for seg in text.split("\n\n")):
            if not paragraph:
                continue
            normalized = " ".join(paragraph.split())
            if normalized in seen:
                continue
            seen.add(normalized)
            merged.append(paragraph)
    return "\n\n".join(merged)

def _build_chunk(
    page_idx: int,
    chunk_idx: int,
    text: str,
    page_images: Dict[int, List[str]],
    ocr_backends: List[str],
    page_image_path: str | None,
    region_hint: str | None,
    layout_metadata: Dict[str, object] | None = None,
    ocr_consensus: float = 1.0,
) -> Chunk:
    page_assets = page_images.get(page_idx, [])
    region_type, region_metadata = classify_region(
        text,
        bool(page_assets),
        figure_hint=(region_hint == "figure"),
    )
    if region_hint and region_type == "text":
        region_type = region_hint
        region_metadata["region_type"] = region_hint
    if layout_metadata:
        hint = layout_metadata.get("tag")
        if hint:
            region_type = hint
            region_metadata["region_type"] = hint
    primary_backend = ocr_backends[0] if ocr_backends else "none"
    metadata: Dict[str, object] = {
        "ocr_backend": primary_backend,
        "ocr_backends": ocr_backends,
        "ocr_consensus": round(float(ocr_consensus), 3),
        "ocr_multi_pass": len(ocr_backends),
        "paragraphs": text.count("\n") + 1,
        "page_image": page_image_path,
        "region_type": region_type,
    }
    metadata.update(region_metadata)
    if layout_metadata:
        for key, value in layout_metadata.items():
            if key == "text":
                continue
            metadata.setdefault(key, value)
    if "layout_confidence" in metadata and metadata["layout_confidence"] is not None:
        try:
            metadata["layout_confidence"] = round(float(metadata["layout_confidence"]), 3)
        except (TypeError, ValueError):
            pass
    metadata["image_refs"] = page_assets if region_type == "figure" else []
    metadata.update(noise_metrics(text))
    branch_id = None
    if layout_metadata:
        branch_id = layout_metadata.get("branch_id")
    branch_provenance: Dict[str, object] = {"primary": "ocr"}
    if branch_id:
        branch_provenance["vision"] = {"branch_id": branch_id}
        metadata["vision_branch_id"] = branch_id
    metadata["branch_provenance"] = branch_provenance
    return Chunk(
        chunk_id=f"page{page_idx + 1:03d}_{chunk_idx:02d}",
        page=page_idx + 1,
        text=text,
        images=page_assets if region_type == "figure" else [],
        metadata=metadata,
    )

def chunk_text(
    pages: Sequence[str],
    page_images: Dict[int, List[str]],
    chunk_chars: int,
    ocr_content: Dict[int, Dict[str, str]],
    page_image_paths: Dict[int, str],
    semantic_chunker: SemanticChunker | None,
    layout_segments: Dict[int, List[Any]] | None = None,
) -> List[Chunk]:
    chunks: List[Chunk] = []
    figure_fingerprints: Set[Tuple[str, ...]] = set()
    prev_embedding = None
    buffer_sentence_count = 0
    segments_map = layout_segments or {}

    for page_idx, raw_text in enumerate(pages):
        text_sources: OrderedDict[str, str] = OrderedDict()
        base_text = raw_text.strip()
        if base_text:
            text_sources["pypdf"] = base_text
        
        # Merge with OCR content
        page_ocr = ocr_content.get(page_idx, {})
        for backend, transcript in page_ocr.items():
            if backend not in text_sources:
                text_sources[backend] = transcript
        
        page_image_path = page_image_paths.get(page_idx)
        
        page_consensus = 0.0
        if not text_sources:
            text = f"[ocr-missing page={page_idx + 1}]"
            backends_used = ["none"]
        else:
            merged = merge_text_sources(list(text_sources.values()))
            text = merged or "\n\n".join(text_sources.values())
            backends_used = list(text_sources.keys())
            page_consensus = ocr_consensus_score(list(text_sources.values()))
            
        base_paragraphs = [para for para in text.split("\n\n") if para.strip()]
        if not base_paragraphs:
            base_paragraphs = [text]
            
        structured_records: List[Tuple[str, str | None, Dict[str, object]]] = []
        seen_normalized: Set[str] = set()
        
        # Process layout regions first
        for region in sorted(segments_map.get(page_idx, []), key=lambda seg: seg.order):
            normalized = " ".join(region.text.split())
            if not normalized:
                continue
            seen_normalized.add(normalized)
            meta = dict(region.extras) if hasattr(region, "extras") else {}
            if isinstance(region, dict): # Fallback if dict
                 meta = region.get("extras", {})
                 meta["tag"] = region.get("tag")
                 text_val = region.get("text", "")
            else:
                 meta["tag"] = region.tag
                 text_val = region.text
            structured_records.append((text_val, meta["tag"], meta))
            
        for para in base_paragraphs:
            normalized = " ".join(para.split())
            if normalized and normalized in seen_normalized:
                continue
            structured_records.append((para, None, {}))
            
        if not structured_records:
            structured_records = [(text, None, {})]
            
        buffer: List[str] = []
        buffer_metadata: Dict[str, object] = {}
        buffer_region = None
        current_len = 0
        chunk_idx = 0
        
        def emit_chunk() -> None:
            nonlocal buffer, buffer_metadata, buffer_region, current_len, chunk_idx, buffer_sentence_count, figure_fingerprints
            if not buffer:
                return
            combined = "\n\n".join(buffer)
            layout_meta = buffer_metadata.copy() if buffer_metadata else None
            chunk = _build_chunk(
                page_idx,
                chunk_idx,
                combined,
                page_images,
                backends_used,
                page_image_path,
                buffer_region,
                layout_meta,
                page_consensus,
            )
            
            appended = True
            if chunk.metadata.get("region_type") == "figure" and chunk.images:
                fingerprint = tuple(sorted(chunk.images))
                if fingerprint in figure_fingerprints:
                    appended = False
                else:
                    figure_fingerprints.add(fingerprint)
            if appended:
                chunks.append(chunk)
            
            buffer = []
            buffer_metadata = {}
            current_len = 0
            chunk_idx += 1
            buffer_sentence_count = 0
            buffer_region = None

        for para, hint_tag, extra_meta in structured_records:
            para = para.strip()
            if not para:
                continue
            region = hint_tag or detect_paragraph_region(para)
            para_len = len(para)
            flush = False
            embedding = semantic_chunker.embed(para) if semantic_chunker else None
            sentence_count = semantic_chunker.sentence_count(para) if semantic_chunker else max(1, para.count("."))
            
            if not buffer_region:
                buffer_region = region
            if region != buffer_region:
                flush = True
            if region in {"heading", "question", "answer", "figure", "table", "formula"} and buffer:
                flush = True
            if current_len + para_len > chunk_chars and buffer:
                flush = True
            semantic_break = (
                semantic_chunker.should_break(prev_embedding, embedding, buffer_sentence_count)
                if semantic_chunker and buffer
                else False
            )
            if semantic_break:
                flush = True
                
            if flush and buffer:
                emit_chunk()
            
            buffer.append(para)
            current_len += para_len
            if extra_meta:
                for key, value in extra_meta.items():
                    if value is None or key == "text":
                        continue
                    buffer_metadata.setdefault(key, value)
            if hint_tag:
                buffer_region = hint_tag
                buffer_metadata.setdefault("tag", hint_tag)
            else:
                buffer_region = region
            buffer_sentence_count += sentence_count or 1
            if embedding is not None:
                prev_embedding = embedding
                
        if buffer:
            emit_chunk()
            
    for idx, chunk in enumerate(chunks):
        prev_region = chunks[idx - 1].metadata.get("region_type") if idx > 0 else None
        next_region = chunks[idx + 1].metadata.get("region_type") if idx + 1 < len(chunks) else None
        chunk.metadata["context"] = {
            "prev_region": prev_region,
            "next_region": next_region,
            "prev_chunk_id": chunks[idx - 1].chunk_id if idx > 0 else None,
            "next_chunk_id": chunks[idx + 1].chunk_id if idx + 1 < len(chunks) else None,
        }
    return chunks
