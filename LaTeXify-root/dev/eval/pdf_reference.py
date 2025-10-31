# dev/eval/pdf_reference.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
import fitz  # PyMuPDF

@dataclass
class PageRef:
    text: str
    blocks: List[Dict[str, Any]]  # raw blocks with bbox and text

def extract_pdf_reference(pdf_path: str) -> List[PageRef]:
    doc = fitz.open(pdf_path)
    out: List[PageRef] = []
    for p in doc:
        # text (layout-preserving to a degree)
        text = p.get_text("text")
        # blocks: [(x0, y0, x1, y1, "text", block_no, block_type, ...)]
        blocks_raw = p.get_text("blocks")
        blocks = []
        for b in blocks_raw:
            blocks.append({
                "bbox": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                "text": (b[4] or "").strip()
            })
        out.append(PageRef(text=text, blocks=blocks))
    return out
