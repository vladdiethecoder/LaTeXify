"""Docling-powered offline ingestion for PDF → structured blocks."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

LOGGER = logging.getLogger(__name__)


def docling_ingest_blocks(pdf_path: Path, *, docling_options: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    """
    Run Docling layout + OCR to produce structured blocks.
    Returns a list of dicts with keys: text, bbox, page_idx, type.
    """
    docling_options = docling_options or {}
    try:
        from docling.document_converter import DocumentConverter, PdfFormatOption  # type: ignore
        from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractOcrOptions  # type: ignore
        from docling.datamodel.base_models import InputFormat
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Docling is not installed; please install docling[vlm,tesserocr].") from exc

    opts = PdfPipelineOptions()
    opts.do_ocr = bool(docling_options.get("do_ocr", True))
    # opts.do_layout_analysis = True # This might not be a direct flag on PdfPipelineOptions, usually implied or under layout_options
    
    # Check if pipeline_name maps to something in PdfPipelineOptions or if we need to configure accelerator/pipeline
    # For now, let's assume default behavior is what we want, or set specific flags if known.
    # The 'pipeline' string option might be for the older API or specific config.
    # vlm_model might be part of accelerator_options or similar.
    
    # If the user passed 'pipeline'="vlm", we might need to adjust how we set it.
    # But based on the fields list, we don't see 'pipeline'. 
    # We do see 'accelerator_options'.
    
    # Tesseract options if available
    if hasattr(opts, "ocr_options"):
        try:
            opts.ocr_options = TesseractOcrOptions()
        except Exception:  # pragma: no cover - defensive
            pass

    format_options = {
        InputFormat.PDF: PdfFormatOption(pipeline_options=opts)
    }
    converter = DocumentConverter(format_options=format_options)
    LOGGER.info("Docling: starting conversion for %s", pdf_path)
    result = converter.convert(str(pdf_path))
    document = getattr(result, "document", None) or result

    payload = None
    try:
        if hasattr(document, "export_to_dict"):
            payload = document.export_to_dict()
        else:
            payload = document.export_json()  # type: ignore[attr-defined]
    except Exception:
        try:
            payload = document.export_to_markdown()  # type: ignore[attr-defined]
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"Docling export failed: {exc}") from exc

    blocks: List[Dict[str, Any]] = []
    try:
        blocks = _parse_docling_payload(payload)
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("Docling JSON parsing failed (%s). Falling back to markdown split.", exc)
        text = payload if isinstance(payload, str) else ""
        blocks = _markdown_to_blocks(text)

    LOGGER.info("Docling: extracted %d blocks", len(blocks))
    return blocks


# Internal helpers ---------------------------------------------------------


def _parse_docling_payload(payload: Any) -> List[Dict[str, Any]]:
    """Parse Docling JSON/dict payload into block dicts."""
    if payload is None:
        return []
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:
            return _markdown_to_blocks(payload)
    if not isinstance(payload, dict):
        return []

    # Docling v2 support
    if "texts" in payload:
        return _parse_docling_v2_payload(payload)

    pages = payload.get("pages") or []
    # Handle Docling v2 beta where pages might be a dict
    if isinstance(pages, dict):
        page_list = []
        sorted_keys = sorted(pages.keys(), key=lambda k: int(k) if str(k).isdigit() else k)
        for k in sorted_keys:
            p = pages[k]
            if isinstance(p, dict):
                if "number" not in p and str(k).isdigit():
                    p["number"] = int(k)
                page_list.append(p)
        pages = page_list

    blocks: List[Dict[str, Any]] = []
    for idx, page in enumerate(pages):
        page_idx = int(page.get("number", idx + 1)) - 1
        # Elements container
        elements = page.get("elements") or []
        blocks.extend(_coerce_elements(elements, page_idx))
        # Some Docling dumps nest content under typed keys
        for key, entries in page.items():
            if key in {"number", "page", "width", "height", "elements"}:
                continue
            if isinstance(entries, list):
                blocks.extend(_coerce_elements(entries, page_idx, hinted_type=key))
    return blocks


def _parse_docling_v2_payload(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parse flattened Docling v2 dict."""
    blocks = []
    # Collect items from all relevant lists
    for kind in ["texts", "tables", "pictures", "equations", "key_value_items", "form_items"]:
        if kind not in payload:
            continue
        for item in payload[kind]:
            text = item.get("text", "")
            label = item.get("label", "text")
            provs = item.get("prov", [])
            if not provs:
                # Items without provenance (e.g. root metadata) usually skipped
                continue
            
            # Use first provenance item for location
            prov = provs[0]
            page_no = prov.get("page_no", 1)
            page_idx = page_no - 1
            
            bbox_data = prov.get("bbox")
            bbox = None
            if bbox_data:
                # Docling v2 bbox is {l, t, r, b, coord_origin='BOTTOMLEFT'} usually
                # Convert to [x0, y0, x1, y1]
                # Assuming BOTTOMLEFT origin: l=x0, b=y0, r=x1, t=y1
                l = bbox_data.get("l", 0.0)
                r = bbox_data.get("r", 0.0)
                t = bbox_data.get("t", 0.0)
                b = bbox_data.get("b", 0.0)
                bbox = (l, b, r, t)
            
            blocks.append({
                "text": text,
                "bbox": bbox,
                "page_idx": page_idx,
                "type": _refine_type(label, item),
                "metadata": {"raw_type": label}
            })
            
    # Sort: Page ASC, then Top-to-Bottom.
    # If origin is Bottom-Left, Top has higher Y. So sort Y descending.
    blocks.sort(key=lambda b: (
        b["page_idx"],
        -1 * (b["bbox"][3] if b["bbox"] else 0), # Top (y1) descending
        b["bbox"][0] if b["bbox"] else 0         # Left (x0) ascending
    ))
    return blocks



def _coerce_elements(entries: List[Dict[str, Any]], page_idx: int, hinted_type: str | None = None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        text = (
            entry.get("text")
            or entry.get("content")
            or entry.get("value")
            or entry.get("markdown")
            or ""
        )
        bbox = _coerce_bbox(entry.get("bbox") or entry.get("bounding_box") or entry.get("box") or entry.get("quad"))
        raw_type = (entry.get("type") or hinted_type or "text").lower()
        block_type = _refine_type(raw_type, entry)
        out.append(
            {
                "text": text,
                "bbox": bbox,
                "page_idx": page_idx,
                "type": block_type,
                "metadata": {"raw_type": raw_type},
            }
        )
    return out


def _coerce_bbox(bbox: Any) -> Tuple[float, float, float, float] | None:
    if bbox is None:
        return None
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        try:
            return tuple(float(v) for v in bbox[:4])  # type: ignore[return-value]
        except Exception:
            return None
    if isinstance(bbox, dict):
        keys = ("x0", "y0", "x1", "y1")
        if all(k in bbox for k in keys):
            try:
                return tuple(float(bbox[k]) for k in keys)  # type: ignore[return-value]
            except Exception:
                return None
    return None


def _refine_type(raw: str, entry: Dict[str, Any]) -> str:
    raw = raw.lower()
    if "table" in raw:
        return "table"
    if "figure" in raw or "image" in raw or "img" in raw:
        return "figure"
    if "equation" in raw or "formula" in raw or "math" in raw:
        return "equation"
    if "heading" in raw or raw.startswith("h") and entry.get("level"):
        return "heading"
    return "text"


def _markdown_to_blocks(markdown_text: str) -> List[Dict[str, Any]]:
    """Fallback: split markdown into text blocks by blank lines."""
    blocks: List[Dict[str, Any]] = []
    if not markdown_text:
        return blocks
    paragraphs = [p.strip() for p in markdown_text.split("\n\n") if p.strip()]
    for idx, para in enumerate(paragraphs):
        blocks.append({"text": para, "bbox": None, "page_idx": 0, "type": "text"})
    return blocks


__all__ = ["docling_ingest_blocks"]
