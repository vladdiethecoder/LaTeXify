# scripts/ocr_ensemble_test.py
from __future__ import annotations
import argparse, json, os, sys, time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from dev.utils.pdf_to_images import pdf_to_images

RUN_STAMP = time.strftime("%Y-%m-%dT%H-%M-%S")
RUN_DIR = Path("dev/runs") / RUN_STAMP

@dataclass
class WorkerSpec:
    import_path: str   # e.g., "dev.ocr_backends.qwen2vl_ocr2b"
    cls_name: str      # "Backend"
    vis_gpu: str       # which single GPU id to expose (e.g., "0" or "1")
    dtype: str         # "auto" | "fp16" | "bf16"

# Map CLI name -> module path
BACKEND_REGISTRY = {
    "qwen2-vl-ocr-2b": ("dev.ocr_backends.qwen2vl_ocr2b", "Backend"),
    "nanonets-ocr2-3b": ("dev.ocr_backends.nanonets_ocr2", "Backend"),
    "nanonets-ocr-s": ("dev.ocr_backends.nanonets_s", "Backend"),
    # "dots-ocr": ("dev.ocr_backends.dots_ocr", "Backend"),
}

def _spawn_backend_worker(spec: WorkerSpec, pages_dir: Path) -> List[Dict]:
    """
    Import the backend with CUDA_VISIBLE_DEVICES set *before* import so torch sees the GPU.
    """
    # set per-worker env first
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = spec.vis_gpu
    # clean up anything that hints CPU fallbacks
    env.pop("ACCELERATE_USE_CPU", None)
    env.pop("HIP_VISIBLE_DEVICES", None)

    # Inject env into this process for import-time side effects
    os.environ.update(env)

    # Import after setting visible device
    mod = __import__(spec.import_path, fromlist=[spec.cls_name])
    Backend = getattr(mod, spec.cls_name)

    be = Backend(dtype=spec.dtype)
    results: List[Dict] = []

    for page_png in sorted(pages_dir.glob("page-*.png")):
        rec = {"model": be.name, "page": page_png.name}
        try:
            r = be.recognize_page(str(page_png), page=1)
            text_md = (r.text_md or "").strip()
            rec["text_len"] = len(text_md)

            # Persist per-(model,page) markdown
            out_dir = RUN_DIR / "outputs" / be.name
            out_dir.mkdir(parents=True, exist_ok=True)
            out_md = out_dir / (page_png.name.replace(".png", ".md"))
            out_md.write_text(text_md, encoding="utf-8")
            rec["out_md"] = str(out_md)

            if getattr(r, "blocks", None):
                blocks_path = out_md.with_suffix(".blocks.json")
                blocks_path.write_text(json.dumps(r.blocks, ensure_ascii=False, indent=2), encoding="utf-8")
                rec["blocks_json"] = str(blocks_path)

            print(f"{be.name:23s} page={page_png.name} len={rec.get('text_len', 0):5d}")
        except KeyboardInterrupt:
            raise
        except Exception as e:
            rec["error"] = str(e)
            print(f"{be.name:23s} FAILED on {page_png}: {e}")
        results.append(rec)
    return results

def _parse_device_map(arg: str, backends: List[str]) -> Dict[str, str]:
    """
    'qwen2-vl-ocr-2b:0,nanonets-ocr2-3b:1,nanonets-ocr-s:1' -> {name: gpu_id}
    """
    if not arg:
        # default: everything on GPU 0
        return {b: "0" for b in backends}
    d: Dict[str, str] = {}
    for pair in arg.split(","):
        name, gpu = pair.split(":")
        d[name.strip()] = gpu.strip()
    # sanity for missing
    for b in backends:
        d.setdefault(b, "0")
    return d

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pdf", type=str, default=None)
    p.add_argument("--dpi", type=int, default=400)
    p.add_argument("--dtype", type=str, default="auto", choices=["auto", "fp16", "bf16"])
    p.add_argument("--backends", type=str, default="qwen2-vl-ocr-2b,nanonets-ocr2-3b,nanonets-ocr-s")
    p.add_argument("--device_map", type=str, default="")
    p.add_argument("--concurrent", type=int, default=2)  # reserved; we run serially per GPU id
    args = p.parse_args()

    # ---- HF auth check (needed for Qwen models) ----
    # If gated, user must provide HF token. We fail fast with a clear error.
    if "Qwen/Qwen2.5-VL-2B-Instruct" in (os.getenv("QWEN_VL_2B_ID") or ""):
        if not (os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_HOME")):
            print("[warn] Qwen repo may be gated. Set HF token via `export HF_TOKEN=...` or `huggingface-cli login`.")
    # ------------------------------------------------

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    dev_map = _parse_device_map(args.device_map, backends)
    print(f"[info] GPUs available (CUDA_VISIBLE_DEVICES): {os.environ.get('CUDA_VISIBLE_DEVICES', '(not set)')}")
    print(f"[info] Backends: {', '.join(backends)}")
    print(f"[info] Device map: {dev_map}")
    print(f"[info] Concurrency: {args.concurrent} (one worker per GPU)")

    # Prepare run dirs
    (RUN_DIR / "pages").mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "outputs").mkdir(parents=True, exist_ok=True)

    # Rasterize
    if not args.pdf:
        inbox = Path("data/inbox")
        pdfs = sorted(inbox.glob("*.pdf"))
        if not pdfs:
            raise SystemExit("No PDF given and data/inbox is empty.")
        pdf_path = pdfs[0]
    else:
        pdf_path = Path(args.pdf)

    page_paths = pdf_to_images(pdf_path, out_dir=RUN_DIR / "pages", dpi=args.dpi, prefix="page")
    pages_dir = RUN_DIR / "pages"

    # Run each backend on its assigned GPU (serial per backend, parallelizable externally)
    per_page_index: Dict[str, Dict] = {p.name if isinstance(p, Path) else Path(p).name: {"page": Path(p).name, "results": []}
                                       for p in page_paths}

    for b in backends:
        if b not in BACKEND_REGISTRY:
            print(f"[warn] Unknown backend '{b}', skipping.")
            continue
        mod_path, cls_name = BACKEND_REGISTRY[b]
        spec = WorkerSpec(import_path=mod_path, cls_name=cls_name, vis_gpu=dev_map[b], dtype=args.dtype)
        results = _spawn_backend_worker(spec, pages_dir)
        # Merge results into per_page_index
        for rec in results:
            per_page_index[rec["page"]]["results"].append(rec)

    per_page = [per_page_index[k] for k in sorted(per_page_index.keys())]

    summary = {
        "pdf": str(pdf_path),
        "pages": len(per_page),
        "run_dir": str(RUN_DIR),
        "per_page": per_page,
        "checks": {
            "text_from_2_plus_models": any(
                sum(1 for r in p["results"] if r.get("text_len", 0) > 0) >= 2 for p in per_page
            ),
            "pages_processed": len(per_page),
        },
    }

    (RUN_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n=== SUMMARY ===\n{json.dumps(summary, ensure_ascii=False, indent=2)}")

if __name__ == "__main__":
    main()
