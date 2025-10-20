# scripts/ocr_ensemble_test.py
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Union

from dev.utils.pdf_to_images import pdf_to_images
from dev.ocr_backends.nanonets_ocr2 import Backend as NN2
from dev.ocr_backends.nanonets_s import Backend as NNS
from dev.ocr_backends.qwen2vl_ocr2b import Backend as Qwen

# -----------------------------------------------------------------------------
# Run scaffolding
# -----------------------------------------------------------------------------
RUN_STAMP = time.strftime("%Y-%m-%dT%H-%M-%S")
RUN_DIR = Path("dev/runs") / RUN_STAMP


def _dump_markdown(run_dir: Path, model: str, page_png: Path, text_md: str) -> Path:
    """
    Save OCR text for this (model, page) into:
      dev/runs/<stamp>/outputs/<model>/<page>.md
    Returns the path written.
    """
    out_dir = run_dir / "outputs" / model
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (page_png.name.replace(".png", ".md"))
    out_path.write_text(text_md or "", encoding="utf-8")
    return out_path


def _load_first_pdf_if_needed(args_pdf: str | None) -> Path:
    """
    Resolve the PDF to process: either the one passed via --pdf or the first
    *.pdf under data/inbox/.
    """
    if args_pdf:
        return Path(args_pdf)
    inbox = Path("data/inbox")
    pdfs = sorted(inbox.glob("*.pdf"))
    if not pdfs:
        raise SystemExit("No PDF in data/inbox/. Pass --pdf <file.pdf>.")
    return pdfs[0]


def _ensure_paths(xs: Union[List[str], List[Path]]) -> List[Path]:
    """Coerce a list of strings/Paths to a list of Path objects."""
    return [p if isinstance(p, Path) else Path(p) for p in xs]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rasterize a PDF to images and run multiple OCR backends; "
                    "dump per-model markdown per page and a summary.json."
    )
    parser.add_argument(
        "--pdf",
        type=str,
        default=None,
        help="Path to PDF. Defaults to first in data/inbox/"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=400,
        help="Rasterization DPI for PDF pages (default: 400)"
    )
    args = parser.parse_args()

    # Prepare run folders
    (RUN_DIR / "pages").mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "outputs").mkdir(parents=True, exist_ok=True)

    # Resolve PDF & rasterize to images
    pdf_path = _load_first_pdf_if_needed(args.pdf)
    page_paths = pdf_to_images(
        pdf_path,
        out_dir=RUN_DIR / "pages",
        dpi=args.dpi,
        prefix="page",
    )
    page_paths = _ensure_paths(page_paths)  # ensure Path objects for .name

    # Backends (DoTS left out in this environment for stability)
    backends = [NN2(), NNS(), Qwen()]

    # Build the top-level summary structure
    summary = {
        "pdf": str(pdf_path),
        "pages": len(page_paths),
        "run_dir": str(RUN_DIR),
        "per_page": [],  # [{page, results:[{model,page,text_len,out_md,blocks_json?,error?}]}]
        "checks": {},
    }

    # Per-page inference + saving
    per_page = []
    for page_png in page_paths:
        page_rec = {"page": page_png.name, "results": []}

        for be in backends:
            rec = {"model": be.name, "page": page_png.name}
            try:
                # OCRResult(model, page, text_md, blocks)
                r = be.recognize_page(str(page_png), page=1)
                text_md = (getattr(r, "text_md", "") or "").strip()

                # Save text now so the evaluator can pick it up later
                out_md = _dump_markdown(RUN_DIR, be.name, page_png, text_md)

                rec["text_len"] = len(text_md)
                rec["out_md"] = str(out_md)

                # If backend returned structured blocks, write a sidecar JSON
                blocks = getattr(r, "blocks", None)
                if blocks:
                    blocks_path = out_md.with_suffix(".blocks.json")
                    blocks_path.write_text(
                        json.dumps(blocks, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    rec["blocks_json"] = str(blocks_path)

                print(f"{be.name:23s} page={page_png.name} len={rec['text_len']:5d}")

            except KeyboardInterrupt:
                raise
            except Exception as e:
                rec["error"] = str(e)
                print(f"{be.name:23s} FAILED on {page_png}: {e}")

            page_rec["results"].append(rec)

        per_page.append(page_rec)

    # Simple checks (the heavy analysis is done in scripts/evaluate_run.py)
    summary["per_page"] = per_page
    summary["checks"] = {
        "text_from_2_plus_models": any(
            sum(1 for r in p["results"] if r.get("text_len", 0) > 0) >= 2
            for p in per_page
        ),
        "pages_processed": len(per_page),
    }

    # Persist summary.json at the run root
    (RUN_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
