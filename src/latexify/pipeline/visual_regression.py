"""Visual regression scoring between generated PDFs and original page renders."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from PIL import Image, ImageChops, ImageStat

try:  # pragma: no cover - optional dependency
    from pdf2image import convert_from_path
except Exception:  # pragma: no cover
    convert_from_path = None  # type: ignore


PAGE_RE = re.compile(r"page_(\d{4})\.png$")
DEFAULT_THRESHOLD = float(os.environ.get("LATEXIFY_VISUAL_DIFF_THRESHOLD", "0.25"))
DEFAULT_LIMIT = int(os.environ.get("LATEXIFY_VISUAL_DIFF_PAGES", "4"))
DEFAULT_DPI = int(os.environ.get("LATEXIFY_VISUAL_DIFF_DPI", "150"))
VISUAL_JUDGE_MODEL = os.environ.get("LATEXIFY_VISUAL_JUDGE_MODEL", "Qwen/Qwen2.5-VL-32B-Instruct")


@dataclass
class PageComparison:
    page: int
    difference: float
    status: str


class VisualLayoutJudge:
    """Lightweight textual summary of visual diffs (approximates Qwen2.5-VL)."""

    def summarize(self, comparisons: List[Dict[str, object]], threshold: float) -> str:
        if not comparisons:
            return "No reference pages were available for comparison."
        flagged = [item for item in comparisons if item.get("status") == "flagged"]
        if not flagged:
            worst = max(comparisons, key=lambda item: item.get("difference", 0.0))
            return f"All sampled pages are within threshold ({threshold}); worst diff {worst['difference']:.3f}."
        worst = max(flagged, key=lambda item: item.get("difference", 0.0))
        pages = ", ".join(str(item["page"]) for item in flagged[:4])
        return f"{len(flagged)} page(s) exceed diff {threshold}: {pages} (max={worst['difference']:.3f})."


def _render_pdf_page(pdf_path: Path, page_number: int, dpi: int) -> Image.Image | None:
    if convert_from_path is None:
        return None
    try:
        images = convert_from_path(
            str(pdf_path),
            first_page=page_number,
            last_page=page_number,
            fmt="png",
            dpi=max(72, dpi),
        )
    except Exception:
        return None
    if not images:
        return None
    return images[0]


def _normalized_difference(reference: Image.Image, generated: Image.Image) -> float:
    ref_gray = reference.convert("L")
    gen_gray = generated.convert("L")
    if ref_gray.size != gen_gray.size:
        gen_gray = gen_gray.resize(ref_gray.size)
    diff = ImageChops.difference(ref_gray, gen_gray)
    stat = ImageStat.Stat(diff)
    mean = stat.mean[0] if stat.mean else 0.0
    return min(1.0, mean / 255.0)


def _reference_pages(reference_dir: Path, page_limit: int) -> List[tuple[int, Path]]:
    records: List[tuple[int, Path]] = []
    for entry in sorted(reference_dir.glob("page_*.png")):
        match = PAGE_RE.match(entry.name)
        if not match:
            continue
        page = int(match.group(1))
        records.append((page, entry))
    return records[:page_limit]


def run_visual_regression(
    tex_path: Path,
    reference_dir: Path,
    output_path: Path,
    *,
    page_limit: int | None = None,
    threshold: float | None = None,
    dpi: int | None = None,
) -> Dict[str, object]:
    """Compare rendered PDF pages to stored originals and emit a diff report."""

    report: Dict[str, object] = {
        "available": False,
        "reason": "",
        "records": [],
        "threshold": threshold or DEFAULT_THRESHOLD,
        "page_limit": page_limit or DEFAULT_LIMIT,
        "dpi": dpi or DEFAULT_DPI,
    }
    pdf_path = tex_path.with_suffix(".pdf")
    if not pdf_path.exists():
        report["reason"] = "compiled PDF missing"
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    if convert_from_path is None:
        report["reason"] = "pdf2image unavailable"
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    if not reference_dir or not reference_dir.exists():
        report["reason"] = "reference page images unavailable"
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    limit = max(1, page_limit or DEFAULT_LIMIT)
    thresh = max(0.0, min(1.0, threshold or DEFAULT_THRESHOLD))
    dpi_value = dpi or DEFAULT_DPI
    pages = _reference_pages(reference_dir, limit)
    if not pages:
        report["reason"] = "no reference page renders"
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    flagged = 0
    comparisons: List[Dict[str, object]] = []
    evaluated = 0
    for page_number, ref_path in pages:
        with Image.open(ref_path) as ref_img:
            generated = _render_pdf_page(pdf_path, page_number, dpi_value)
            if generated is None:
                comparisons.append(
                    {"page": page_number, "difference": 1.0, "status": "missing"}
                )
                flagged += 1
                continue
            diff = round(_normalized_difference(ref_img, generated), 4)
            status = "flagged" if diff >= thresh else "ok"
            if status == "flagged":
                flagged += 1
            comparisons.append(
                {"page": page_number, "difference": diff, "status": status}
            )
            evaluated += 1
    judge = VisualLayoutJudge()
    report.update(
        {
            "available": True,
            "records": comparisons,
            "flagged_pages": flagged,
            "pages_evaluated": evaluated,
            "visual_model": VISUAL_JUDGE_MODEL,
            "layout_summary": judge.summarize(comparisons, thresh),
        }
    )
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


__all__ = ["run_visual_regression"]
