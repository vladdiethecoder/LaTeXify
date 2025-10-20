#!/usr/bin/env python3
import json, os, argparse, pathlib, time, glob
from pathlib import Path
from rapidfuzz.distance import Levenshtein
from dev.ocr_backends.nanonets_ocr2 import Backend as NN2
from dev.ocr_backends.nanonets_s import Backend as NNS
from dev.ocr_backends.dots_ocr import Backend as Dots
from dev.ocr_backends.qwen2vl_ocr2b import Backend as Qwen
from dev.ocr_backends.base import OCRResult
from dev.utils.pdf_to_images import pdf_to_images

IMG = Path("data/inbox/sample_page.png")
RUN_DIR = Path("dev/runs") / time.strftime("%Y-%m-%dT%H-%M-%S")
RUN_DIR.mkdir(parents=True, exist_ok=True)

def find_first_pdf(folder: str) -> str | None:
    matches = sorted(glob.glob(os.path.join(folder, "*.pdf")))
    return matches[0] if matches else None

def save(name, obj):
    Path(RUN_DIR / f"{name}.json").write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def main():
    import argparse, pathlib, os, glob, json, time

    from dev.utils.pdf_to_images import pdf_to_images
    from dev.ocr_backends.nanonets_ocr2 import Backend as NN2
    from dev.ocr_backends.nanonets_s import Backend as NNS
    try:
        from dev.ocr_backends.dots_ocr import Backend as Dots
        HAS_DOTS = True
    except Exception as e:
        print("dots.ocr unavailable (continuing without it):", e)
        HAS_DOTS = False
    from dev.ocr_backends.qwen2vl_ocr2b import Backend as Qwen

    def find_first_pdf(folder: str) -> str | None:
        matches = sorted(glob.glob(os.path.join(folder, "*.pdf")))
        return matches[0] if matches else None

    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", type=str, default=None, help="PDF file or directory containing PDFs")
    ap.add_argument("--image", type=str, default=None, help="Single image to OCR (png/jpg)")
    ap.add_argument("--dpi", type=int, default=400)
    args = ap.parse_args()

    ts = time.strftime("%Y-%m-%dT%H-%M-%S")
    run_dir = pathlib.Path(f"dev/runs/{ts}")
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---------- choose input(s) ----------
    if args.pdf:
        target = args.pdf
        if os.path.isdir(target):
            first_pdf = find_first_pdf(target)
            assert first_pdf, f"No PDFs found in directory: {target}"
            target = first_pdf
        assert os.path.exists(target), f"No such file: {target}"
        img_paths = pdf_to_images(target, out_dir=run_dir / "pages", dpi=args.dpi, prefix="page")
    elif args.image:
        assert os.path.exists(args.image), f"No such file: {args.image}"
        img_paths = [args.image]
    else:
        inbox_pdf = find_first_pdf("data/inbox")
        assert inbox_pdf, "No PDF found. Provide --pdf or place a .pdf in data/inbox/"
        img_paths = pdf_to_images(inbox_pdf, out_dir=run_dir / "pages", dpi=args.dpi, prefix="page")

    # ---------- backends ----------
    backends = [NN2(), NNS(), Qwen()] + ([Dots()] if HAS_DOTS else [])

    per_page = []
    for img in img_paths:
        page_results = []
        for be in backends:
            try:
                r = be.recognize_page(img, page=1)
                page_results.append({"model": be.name, "text": r.text_md, "blocks": len(r.blocks)})
                print(f"{be.name:22s} page={pathlib.Path(img).name:>12s} len={len(r.text_md):5d}")
            except Exception as e:
                print(f"{be.name} FAILED on {img}: {e}")
        per_page.append({"image": img, "results": page_results})

    # ---------- summary checks ----------
    all_texts = [res["text"] for page in per_page for res in page["results"] if "text" in res]
    consensus_len = max((len(t) for t in all_texts), default=0)
    has_structure = any(("\\(" in t or "<table" in t or "\\begin{equation}" in t) for t in all_texts)
    # consider pass if at least 2 distinct models produced text on any page
    models_with_text = {res["model"] for page in per_page for res in page["results"] if len(res.get("text","")) > 0}
    checks = {
        "text_from_2_plus_models": len(models_with_text) >= 2,
        "consensus_len_over_200": consensus_len > 200,
        "has_structure_hint": bool(has_structure),
    }

    summary = {
        "pdf": args.pdf,
        "pages": len(img_paths),
        "run_dir": str(run_dir),
        "checks": checks,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
