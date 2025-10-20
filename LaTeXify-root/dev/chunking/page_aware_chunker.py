# dev/chunking/page_aware_chunker.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
import math

from PIL import Image
from pydantic import BaseModel

from transformers import AutoProcessor, AutoModelForTokenClassification, pipeline

# HURIDOCS pdf document layout analysis (HF pipeline):
# Model card shows token classification outputs per, e.g., paragraphs, headers, tables, figures, etc.
# https://huggingface.co/HURIDOCS/pdf-document-layout-analysis
MODEL_ID = "HURIDOCS/pdf-document-layout-analysis"

# ---- Data structures ----

@dataclass
class Region:
    label: str
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    score: float

@dataclass
class Chunk:
    chunk_id: str
    page: int
    label: str
    bbox: Tuple[int, int, int, int]
    text: str

class ChunkPack(BaseModel):
    pdf_path: str
    run_dir: str
    chunks: List[Dict[str, Any]]

# ---- Core helpers ----

def _iou(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    xi1, yi1 = max(ax1, bx1), max(ay1, by1)
    xi2, yi2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, xi2 - xi1), max(0, yi2 - yi1)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter if (area_a + area_b - inter) > 0 else 1.0
    return inter / union

def _center(b: Tuple[int,int,int,int]) -> Tuple[float,float]:
    x1,y1,x2,y2 = b
    return ((x1+x2)/2.0, (y1+y2)/2.0)

def _closest_region(
    text_bbox: Tuple[int,int,int,int],
    regions: List[Region],
    iou_thresh: float = 0.05
) -> Optional[Region]:
    # Prefer overlap; fallback to nearest center
    best: Optional[Region] = None
    best_score = -1.0
    cx, cy = _center(text_bbox)
    for r in regions:
        ov = _iou(text_bbox, r.bbox)
        if ov >= iou_thresh and ov > best_score:
            best, best_score = r, ov
    if best is not None:
        return best
    # distance fallback
    dmin, dreg = float("inf"), None
    for r in regions:
        rx, ry = _center(r.bbox)
        d = math.hypot(cx - rx, cy - ry)
        if d < dmin:
            dmin, dreg = d, r
    return dreg

# ---- Layout inference ----

def _load_layout_pipeline():
    # HF pipeline for token classification over PDF-page images
    return pipeline(
        task="token-classification",
        model=MODEL_ID,
        aggregation_strategy="simple"  # merge sub-tokens into entity spans
    )

def _infer_regions_on_page(img_path: Path, nlp) -> List[Region]:
    im = Image.open(img_path).convert("RGB")
    # Many layout models return entities with 'entity_group' + 'box'/'bbox' in normalized coords
    out = nlp(im)
    regions: List[Region] = []
    for i, ent in enumerate(out):
        label = ent.get("entity_group") or ent.get("label") or "other"
        score = float(ent.get("score", 0.0))
        # Expect xyxy in pixel coords if available; otherwise map from normalized boxes
        box = ent.get("box") or ent.get("bbox")
        if isinstance(box, dict) and {"xmin","ymin","xmax","ymax"} <= set(box.keys()):
            x1, y1, x2, y2 = int(box["xmin"]), int(box["ymin"]), int(box["xmax"]), int(box["ymax"])
        elif isinstance(box, (list, tuple)) and len(box) == 4:
            x1, y1, x2, y2 = map(int, box)
        else:
            # If only normalized coords exist, estimate via image size
            w, h = im.size
            nx1, ny1 = float(ent.get("start", 0.0)), float(ent.get("top", 0.0))
            nx2, ny2 = float(ent.get("end", 1.0)), float(ent.get("bottom", 1.0))
            x1, y1, x2, y2 = int(nx1*w), int(ny1*h), int(nx2*w), int(ny2*h)
        regions.append(Region(label=label, bbox=(x1,y1,x2,y2), score=score))
    return regions

# ---- Chunk construction ----

def _load_page_text_blocks(run_dir: Path, page_png: Path) -> List[Dict[str, Any]]:
    """
    Reads model outputs dumped by scripts/ocr_ensemble_test.py:
      dev/runs/<stamp>/outputs/<model>/<page>.md
      dev/runs/<stamp>/outputs/<model>/<page>.blocks.json (optional with bboxes)
    Returns a list of "blocks" with approximate bboxes when available.
    """
    blocks: List[Dict[str,Any]] = []
    outputs_dir = run_dir / "outputs"
    for model_dir in outputs_dir.iterdir():
        if not model_dir.is_dir():
            continue
        md_file = model_dir / page_png.name.replace(".png", ".md")
        if not md_file.exists():
            continue
        text = md_file.read_text(encoding="utf-8").strip()
        if not text:
            continue
        # Try sidecar blocks with boxes
        sidecar = md_file.with_suffix(".blocks.json")
        if sidecar.exists():
            try:
                obj = json.loads(sidecar.read_text(encoding="utf-8"))
                for b in obj:
                    t = (b.get("text") or "").strip()
                    bb = b.get("bbox") or b.get("box")
                    if t and isinstance(bb, (list, tuple)) and len(bb) == 4:
                        blocks.append({"text": t, "bbox": tuple(map(int, bb)), "model": model_dir.name})
            except Exception:
                pass
        if not any(b.get("model") == model_dir.name for b in blocks):
            # Fallback: one big block without bbox
            blocks.append({"text": text, "bbox": None, "model": model_dir.name})
    return blocks

def build_page_chunks(run_dir: Path, page_idx: int, page_png: Path, nlp) -> List[Chunk]:
    regions = _infer_regions_on_page(page_png, nlp)
    text_blocks = _load_page_text_blocks(run_dir, page_png)
    chunks: List[Chunk] = []

    # If no explicit bboxes for blocks, we still create chunks per model per page
    # and attach a best-guess region (closest by center == page center).
    page_center_region = None
    if regions:
        # choose largest region as default for bbox-less blocks
        page_center_region = max(regions, key=lambda r: (r.bbox[2]-r.bbox[0])*(r.bbox[3]-r.bbox[1]))

    for bi, tb in enumerate(text_blocks):
        if tb["bbox"] is not None and regions:
            reg = _closest_region(tuple(tb["bbox"]), regions) or page_center_region
        else:
            reg = page_center_region
        label = reg.label if reg else "page"
        bbox = reg.bbox if reg else (0,0,0,0)
        text = tb["text"]
        chunks.append(
            Chunk(
                chunk_id=f"p{page_idx:04d}_b{bi:04d}",
                page=page_idx,
                label=label,
                bbox=bbox,
                text=text,
            )
        )
    return chunks

def build_chunks_for_run(run_dir: Path, pdf_path: Path) -> Path:
    """
    Given a run directory (with /pages and /outputs), produce layout-aware chunks and
    write them to dev/runs/<stamp>/chunks.jsonl
    """
    pages_dir = run_dir / "pages"
    page_paths = sorted(pages_dir.glob("*.png"))
    if not page_paths:
        raise SystemExit(f"No page images in {pages_dir}")

    nlp = _load_layout_pipeline()
    all_chunks: List[Chunk] = []
    for idx, page_png in enumerate(page_paths, start=1):
        page_chunks = build_page_chunks(run_dir, idx, page_png, nlp)
        all_chunks.extend(page_chunks)

    out = run_dir / "chunks.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for ch in all_chunks:
            f.write(json.dumps(asdict(ch), ensure_ascii=False) + "\n")
    return out
