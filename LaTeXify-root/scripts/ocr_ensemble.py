#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR Ensemble runner (no fallbacks, hard-fail on missing engines).

Engines supported:
  - qwen2vl      : Qwen/Qwen2-VL-* via transformers + qwen-vl-utils
  - nanonets     : Nanonets OCR2/3B (Qwen2.5-VL family) via transformers
  - tesseract    : pytesseract (CPU)
  - paddle       : PaddleOCR (optional)

Input:
  --pdf PATH            : PDF to OCR (preferred)
  --images-dir PATH     : directory of images instead of PDF (alternative)

Output:
  --out PATH            : JSONL, one record per (page, engine)
                          {"page": N, "engine": "qwen2vl", "text": "...", "secs": 0.42}

Strictness:
  * If any engine listed in --engines cannot be loaded, exit(2)
  * If no pages produced output, exit(3)

Notes:
  * pdf2image is used first (requires poppler-utils). Fallback to PyMuPDF (fitz).
  * For Qwen2-VL usage see:
      - https://qwen.readthedocs.io/en/v2.0/getting_started/quickstart.html
      - https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct (API family docs)
  * For Nanonets OCR2-3B model card:
      - https://huggingface.co/NanoNets/ocr-2_3b_instruct
  * For pdf2image / PyMuPDF / Tesseract:
      - pdf2image: pip install pdf2image (poppler required)
      - PyMuPDF: pip install PyMuPDF
      - pytesseract: pip install pytesseract (dnf install tesseract tesseract-langpack-eng)
"""

from __future__ import annotations
<<<<<<< ours
=======

>>>>>>> theirs
import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
<<<<<<< ours
from typing import Iterable, List, Optional, Tuple
=======
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
>>>>>>> theirs

# ---------- Utilities: PDF -> PIL Images ----------
def pdf_to_images(pdf_path: Path, dpi: int = 300) -> List["Image.Image"]:
    """Convert a PDF into a list of PIL images; prefer pdf2image, fallback to PyMuPDF."""
    try:
        from pdf2image import convert_from_path  # requires poppler
        return convert_from_path(str(pdf_path), dpi=dpi)
    except Exception:
        # Fallback: PyMuPDF / fitz
        try:
            import fitz  # PyMuPDF
            imgs = []
            doc = fitz.open(str(pdf_path))
            zoom = dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            for page in doc:
                pix = page.get_pixmap(matrix=mat, alpha=False)
                from PIL import Image
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                imgs.append(img)
            return imgs
        except Exception as e:
            raise RuntimeError(f"Failed to render PDF (need pdf2image+poppler or PyMuPDF): {e}")

<<<<<<< ours
def load_images_from_dir(img_dir: Path) -> List["Image.Image"]:
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
    files = sorted([p for p in img_dir.glob("**/*") if p.suffix.lower() in exts])
    if not files:
        raise RuntimeError(f"No images found under {img_dir}")
    from PIL import Image
    return [Image.open(p).convert("RGB") for p in files]
=======
RUN_STAMP = time.strftime("%Y-%m-%dT%H-%M-%S")
>>>>>>> theirs

# ---------- Engines ----------
@dataclass
<<<<<<< ours
class EngineSpec:
    name: str
    required: bool
    model_dir: Optional[Path] = None

class OCREngine:
    def __init__(self, device: str = "cuda"):
        self.device = device

    def name(self) -> str:
        raise NotImplementedError

    def ensure_ready(self) -> None:
        """Load heavy resources here; must raise on failure."""
        raise NotImplementedError

    def run_page(self, img, page_no: int) -> Tuple[str, float]:
        """Return (text, seconds) for a single page image; raise on failure."""
        raise NotImplementedError

# --- Qwen2-VL / Nanonets (transformers) ---
class QwenVLEngine(OCREngine):
    def __init__(self, model_dir: Path, device: str = "cuda"):
        super().__init__(device)
        self.model_dir = str(model_dir)
        self._model = None
        self._processor = None

    def name(self) -> str:
        return "qwen2vl"

    def ensure_ready(self) -> None:
        try:
            import torch
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
            from qwen_vl_utils import process_vision_info
            self._torch = torch
            self._pvi = process_vision_info
            self._processor = AutoProcessor.from_pretrained(self.model_dir, trust_remote_code=True)
            self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_dir,
                torch_dtype="auto",
                device_map="auto",
            )
        except Exception as e:
            raise RuntimeError(f"Qwen2-VL engine load failed: {e}")

    def run_page(self, img, page_no: int) -> Tuple[str, float]:
        torch = self._torch
        process_vision_info = self._pvi
        t0 = time.time()
        messages = [
            {"role": "system", "content": "You are an OCR engine. Extract only the readable text from the page. Keep math inline, avoid extra commentary."},
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": "Return plain text with line breaks. No markdown headings, no explanations."}
            ]},
        ]
        prompt = self._processor.apply_chat_template(messages, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = self._processor(text=prompt, images=image_inputs, return_tensors="pt").to(self._model.device)
        with torch.inference_mode():
            gen = self._model.generate(**inputs, max_new_tokens=1024, do_sample=False)
        # strip the prompt tokens
        out_tokens = gen[:, inputs["input_ids"].shape[-1]:]
        text = self._processor.batch_decode(out_tokens, skip_special_tokens=True)[0]
        return text.strip(), time.time() - t0

class NanonetsVLEngine(QwenVLEngine):
    def name(self) -> str:
        return "nanonets"

# --- Tesseract (CPU) ---
class TesseractEngine(OCREngine):
    def name(self) -> str:
        return "tesseract"

    def ensure_ready(self) -> None:
        try:
            import pytesseract  # noqa
            from PIL import Image  # noqa
        except Exception as e:
            raise RuntimeError(f"pytesseract not available: {e}")

    def run_page(self, img, page_no: int) -> Tuple[str, float]:
        import pytesseract
        t0 = time.time()
        txt = pytesseract.image_to_string(img)
        return txt.strip(), time.time() - t0

# --- PaddleOCR (optional) ---
class PaddleOCREngine(OCREngine):
    def name(self) -> str:
        return "paddle"

    def ensure_ready(self) -> None:
        try:
            from paddleocr import PaddleOCR  # noqa
        except Exception as e:
            raise RuntimeError(f"PaddleOCR not available: {e}")
        # lazy init in run_page

    def run_page(self, img, page_no: int) -> Tuple[str, float]:
        from paddleocr import PaddleOCR
        import numpy as np
        t0 = time.time()
        # english; adjust as needed
        ocr = PaddleOCR(use_angle_cls=True, lang="en")
        arr = np.array(img)[:, :, ::-1]  # RGB->BGR
        res = ocr.ocr(arr, cls=True)
        lines = []
        for page in res:
            for (_box, (_txt, _score)) in page:
                lines.append(_txt)
        return "\n".join(lines).strip(), time.time() - t0

# ---------- CLI / Main ----------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="OCR Ensemble (strict)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pdf", type=str, help="Input PDF")
    src.add_argument("--images-dir", type=str, help="Directory of input page images")

    ap.add_argument("--out", type=str, required=True, help="Output JSONL file")
    ap.add_argument("--engines", type=str, default="qwen2vl,nanonets,tesseract",
                    help="Comma list: qwen2vl,nanonets,tesseract,paddle")

    ap.add_argument("--qwen-model", type=str, default=None,
                    help="Path to local Qwen2-VL-* Instruct model dir")
    ap.add_argument("--nanonets-model", type=str, default=None,
                    help="Path to local Nanonets OCR2/3B Instruct model dir")

    ap.add_argument("--dpi", type=int, default=300, help="Rasterization DPI for PDF")
    ap.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    ap.add_argument("--page-range", type=str, default=None,
                    help="e.g., 1-3,6 to subset pages (1-based)")

    return ap.parse_args()

def parse_page_range(spec: Optional[str], n_pages: int) -> List[int]:
    if not spec:
        return list(range(1, n_pages + 1))
    out: List[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            a, b = int(a), int(b)
            out.extend(list(range(a, b + 1)))
        else:
            out.append(int(chunk))
    out = [p for p in out if 1 <= p <= n_pages]
    return sorted(set(out))

def main() -> None:
    args = parse_args()
    engines_req = [e.strip().lower() for e in args.engines.split(",") if e.strip()]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load pages
    if args.pdf:
        pdf_path = Path(args.pdf)
        pages = pdf_to_images(pdf_path, dpi=args.dpi)
    else:
        pages = load_images_from_dir(Path(args.images_dir))
    if not pages:
        print("No pages produced", file=sys.stderr)
        sys.exit(3)

    # Build engine instances based on CLI
    engine_objs: List[OCREngine] = []
    failures: List[str] = []

    def need(name: str) -> bool:
        return name in engines_req

    # Qwen2-VL
    if need("qwen2vl"):
        if not args.qwen_model:
            failures.append("qwen2vl requested but --qwen-model not provided")
        else:
            engine_objs.append(QwenVLEngine(Path(args.qwen_model), device=args.device))

    # Nanonets
    if need("nanonets"):
        if not args.nanonets_model:
            failures.append("nanonets requested but --nanonets-model not provided")
        else:
            engine_objs.append(NanonetsVLEngine(Path(args.nanonets_model), device=args.device))

    # Tesseract
    if need("tesseract"):
        engine_objs.append(TesseractEngine(device="cpu"))

    # PaddleOCR
    if need("paddle"):
        engine_objs.append(PaddleOCREngine(device="cpu"))

    if failures:
        for f in failures:
            print(f"[ocr] ERROR: {f}", file=sys.stderr)
        sys.exit(2)

    # Ensure engines are ready
    for eng in engine_objs:
        eng.ensure_ready()

    # Iterate pages
    import traceback
    n = len(pages)
    keep_pages = parse_page_range(args.page_range, n)
    wrote = 0
    with out_path.open("w", encoding="utf-8") as f:
        for idx, pno in enumerate(keep_pages, start=1):
            img = pages[pno - 1]
            for eng in engine_objs:
                try:
                    text, secs = eng.run_page(img, pno)
                    rec = {"page": pno, "engine": eng.name(), "text": text, "secs": round(secs, 3)}
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    wrote += 1
                except Exception:
                    tb = traceback.format_exc()
                    print(f"[ocr] {eng.name()} failed on page {pno}:\n{tb}", file=sys.stderr)
                    # Strict: fail immediately if any engine crashes on a page
                    sys.exit(4)

    if wrote == 0:
        print("[ocr] No output records written", file=sys.stderr)
        sys.exit(3)

    print(f"[ocr] Wrote {wrote} records to {out_path}")
=======
class BackendSpec:
    name: str  # registry key, e.g. qwen2-vl-ocr-2b
    module: str  # python module path
    cls: str  # class name
    device: Optional[str]  # which GPU id in this worker (string)
    dtype: str  # fp16/bf16

    def instantiate(self):
        mod = __import__(self.module, fromlist=[self.cls])
        Backend = getattr(mod, self.cls)
        return Backend(dtype=self.dtype)

# Registry of available backends
REGISTRY: Dict[str, Dict[str, str]] = {
    "qwen2-vl-ocr-2b": {"module": "dev.ocr_backends.qwen2vl_ocr2b", "cls": "Backend"},
    "nanonets-ocr2-3b": {"module": "dev.ocr_backends.nanonets_ocr2", "cls": "Backend"},
    "nanonets-ocr-s": {"module": "dev.ocr_backends.nanonets_s", "cls": "Backend"},
    # Extend with paddleocr later: "paddleocr": {...}
}

def _parse_device_map(s: Optional[str], backends: List[str]) -> Dict[str, Optional[str]]:
    """
    Parse "name:gpu,name2:gpu" into dict; accepts numeric or string ids.
    Missing entries -> None (worker will still have isolated CUDA_VISIBLE_DEVICES).
    """
    d: Dict[str, Optional[str]] = {b: None for b in backends}
    if not s:
        return d
    for kv in s.split(","):
        if ":" not in kv:
            continue
        k, v = kv.split(":", 1)
        k = k.strip()
        v = v.strip()
        if k in d:
            d[k] = v
    return d

def _spawn_backend_worker(spec: BackendSpec, pages_dir: Path, run_dir: Path) -> List[Dict[str, Any]]:
    """
    Single-process worker:
    - isolates the desired GPU via CUDA_VISIBLE_DEVICES
    - instantiates backend with dtype
    - runs all pages sequentially for that backend
    """
    # Isolate GPU in this process
    prev_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    if spec.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(spec.device)
    # Safe defaults for FP16 & Flash-SDP/FA2 (global)
    try:
        import torch

        torch.set_float32_matmul_precision("medium")
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            from torch.backends.cuda import sdp_kernel

            sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=True)
        except Exception:
            pass
    except Exception:
        pass

    try:
        be = spec.instantiate()
    except ModuleNotFoundError as exc:
        missing = exc.name or str(exc)
        print(
            f"[warn] {spec.name}: missing optional dependency '{missing}'. Skipping backend.",
            file=sys.stderr,
        )
        return []
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[warn] {spec.name}: failed to initialize ({exc}). Skipping backend.", file=sys.stderr)
        return []
    results: List[Dict[str, Any]] = []
    try:
        for page_png in sorted((pages_dir).glob("*.png")):
            rec: Dict[str, Any] = {"model": spec.name, "page": page_png.name}
            try:
                r = be.recognize_page(str(page_png), page=1)
                text_md = (r.text_md or "").strip()
                rec["text_len"] = len(text_md)
                out_md = _dump_markdown(run_dir, spec.name, page_png, text_md)
                rec["out_md"] = str(out_md)
                if getattr(r, "blocks", None):
                    blocks_path = out_md.with_suffix(".blocks.json")
                    blocks_path.write_text(
                        json.dumps(r.blocks, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    rec["blocks_json"] = str(blocks_path)
                print(f"{spec.name:23s} page={page_png.name} len={rec.get('text_len', 0):5d}")
            except KeyboardInterrupt:
                raise
            except Exception as e:  # pragma: no cover - backend failure diagnostics
                rec["error"] = str(e)
                print(f"{spec.name:23s} FAILED on {page_png}: {e}")
            results.append(rec)
        return results
    finally:
        if spec.device is not None:
            if prev_cuda is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = prev_cuda

def _dump_markdown(run_dir: Path, model: str, page_png: Path, text_md: str) -> Path:
    out_dir = run_dir / "outputs" / model
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (page_png.name.replace(".png", ".md"))
    out_path.write_text(text_md or "", encoding="utf-8")
    return out_path

def _load_first_pdf_if_needed(args_pdf: Optional[str]) -> Path:
    if args_pdf:
        p = Path(args_pdf)
        if not p.is_file():
            raise SystemExit(f"PDF not found: {p}")
        return p
    inbox = Path("data/inbox")
    pdfs = sorted(inbox.glob("*.pdf"))
    if not pdfs:
        raise SystemExit("No PDF in data/inbox/. Pass --pdf <file.pdf>.")
    return pdfs[0]

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pdf", type=str, default=None)
    p.add_argument("--dpi", type=int, default=400)
    p.add_argument("--dtype", type=str, default="fp16")
    p.add_argument(
        "--backends",
        "--engines",
        dest="backends",
        type=str,
        default="qwen2-vl-ocr-2b,nanonets-ocr2-3b,nanonets-ocr-s",
        help="Comma-separated list of OCR backends to run.",
    )
    p.add_argument("--device_map", type=str, default=None,
                   help="Comma list: name:gpu_id,... (e.g., qwen2-vl-ocr-2b:0,nanonets-ocr2-3b:1)")
    p.add_argument("--concurrent", type=int, default=1)
    p.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Optional directory for intermediate outputs (defaults to dev/runs/<timestamp>).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional JSONL path to store per-backend page results.",
    )
    args = p.parse_args()

    run_dir = args.run_dir or (Path("dev/runs") / RUN_STAMP)
    run_dir = run_dir.resolve()

    # Prepare run folders
    (run_dir / "pages").mkdir(parents=True, exist_ok=True)
    (run_dir / "outputs").mkdir(parents=True, exist_ok=True)
    summary = {
        "pdf": None,
        "pages": 0,
        "run_dir": str(run_dir),
        "per_page": [],
        "checks": {},
    }

    # Resolve PDF & rasterize
    pdf_path = _load_first_pdf_if_needed(args.pdf)
    summary["pdf"] = str(pdf_path)
    page_paths = pdf_to_images(pdf_path, out_dir=run_dir / "pages", dpi=args.dpi, prefix="page")
    summary["pages"] = len(page_paths)

    # Build spec list
    names = [s.strip() for s in args.backends.split(",") if s.strip()]
    devmap = _parse_device_map(args.device_map, names)
    specs: List[BackendSpec] = []
    for name in names:
        reg = REGISTRY.get(name)
        if not reg:
            print(f"[warn] unknown backend: {name}")
            continue
        specs.append(
            BackendSpec(
                name=name,
                module=reg["module"],
                cls=reg["cls"],
                device=devmap.get(name),
                dtype=args.dtype,
            )
        )

    # Serial per-GPU worker model (simple, reliable); spawn per spec
    all_results: List[Dict[str, Any]] = []
    for spec in specs:
        print(
            f"[worker] {spec.name} -> GPU {spec.device if spec.device is not None else '(auto)'} ; dtype={spec.dtype}"
        )
        results = _spawn_backend_worker(spec, run_dir / "pages", run_dir)
        all_results.extend(results)

    # Compute per-page structure for the summary
    per_page = []
    by_page: Dict[str, List[Dict[str, Any]]] = {}
    for rec in all_results:
        by_page.setdefault(rec["page"], []).append(rec)
    for page_png in sorted((run_dir / "pages").glob("*.png")):
        per_page.append({"page": page_png.name, "results": by_page.get(page_png.name, [])})

    summary["per_page"] = per_page
    summary["checks"] = {
        "text_from_2_plus_models": any(
            sum(1 for r in p["results"] if "text_len" in r and r["text_len"] > 0) >= 2 for p in per_page
        ),
        "pages_processed": len(per_page),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as fh:
            for rec in all_results:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[write] {len(all_results)} records → {args.out}")

    print(f"\n=== SUMMARY ===\n{json.dumps(summary, indent=2)}")
>>>>>>> theirs

if __name__ == "__main__":
    main()
