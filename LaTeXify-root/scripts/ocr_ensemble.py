# scripts/ocr_ensemble_test.py
from __future__ import annotations
import argparse, json, time, os, sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

from dev.utils.pdf_to_images import pdf_to_images

RUN_STAMP = time.strftime("%Y-%m-%dT%H-%M-%S")
RUN_DIR = Path("dev/runs") / RUN_STAMP

@dataclass
class BackendSpec:
    name: str           # registry key, e.g. qwen2-vl-ocr-2b
    module: str         # python module path
    cls: str            # class name
    device: str | None  # which GPU id in this worker (string)
    dtype: str          # fp16/bf16
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

def _parse_device_map(s: str | None, backends: List[str]) -> Dict[str, str | None]:
    """
    Parse "name:gpu,name2:gpu" into dict; accepts numeric or string ids.
    Missing entries -> None (worker will still have isolated CUDA_VISIBLE_DEVICES).
    """
    d: Dict[str, str | None] = {b: None for b in backends}
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

def _spawn_backend_worker(spec: BackendSpec, pages_dir: Path) -> List[Dict[str, Any]]:
    """
    Single-process worker:
    - isolates the desired GPU via CUDA_VISIBLE_DEVICES
    - instantiates backend with dtype
    - runs all pages sequentially for that backend
    """
    # Isolate GPU in this process
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

    be = spec.instantiate()
    results: List[Dict[str, Any]] = []
    for page_png in sorted((pages_dir).glob("*.png")):
        rec = {"model": spec.name, "page": page_png.name}
        try:
            r = be.recognize_page(str(page_png), page=1)
            text_md = (r.text_md or "").strip()
            rec["text_len"] = len(text_md)
            out_md = _dump_markdown(RUN_DIR, spec.name, page_png, text_md)
            rec["out_md"] = str(out_md)
            if getattr(r, "blocks", None):
                blocks_path = out_md.with_suffix(".blocks.json")
                blocks_path.write_text(json.dumps(r.blocks, ensure_ascii=False, indent=2), encoding="utf-8")
                rec["blocks_json"] = str(blocks_path)
            print(f"{spec.name:23s} page={page_png.name} len={rec.get('text_len', 0):5d}")
        except KeyboardInterrupt:
            raise
        except Exception as e:
            rec["error"] = str(e)
            print(f"{spec.name:23s} FAILED on {page_png}: {e}")
        results.append(rec)
    return results

def _dump_markdown(run_dir: Path, model: str, page_png: Path, text_md: str) -> Path:
    out_dir = run_dir / "outputs" / model
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (page_png.name.replace(".png", ".md"))
    out_path.write_text(text_md or "", encoding="utf-8")
    return out_path

def _load_first_pdf_if_needed(args_pdf: str | None) -> Path:
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
    p.add_argument("--backends", type=str,
                   default="qwen2-vl-ocr-2b,nanonets-ocr2-3b,nanonets-ocr-s")
    p.add_argument("--device_map", type=str, default=None,
                   help="Comma list: name:gpu_id,... (e.g., qwen2-vl-ocr-2b:0,nanonets-ocr2-3b:1)")
    p.add_argument("--concurrent", type=int, default=1)
    args = p.parse_args()

    # Prepare run folders
    (RUN_DIR / "pages").mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "outputs").mkdir(parents=True, exist_ok=True)
    summary = {
        "pdf": None,
        "pages": 0,
        "run_dir": str(RUN_DIR),
        "per_page": [],
        "checks": {},
    }

    # Resolve PDF & rasterize
    pdf_path = _load_first_pdf_if_needed(args.pdf)
    summary["pdf"] = str(pdf_path)
    page_paths = pdf_to_images(pdf_path, out_dir=RUN_DIR / "pages", dpi=args.dpi, prefix="page")
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
        print(f"[worker] {spec.name} -> GPU {spec.device if spec.device is not None else '(auto)'} ; dtype={spec.dtype}")
        results = _spawn_backend_worker(spec, RUN_DIR / "pages")
        all_results.extend(results)

    # Compute per-page structure for the summary
    per_page = []
    by_page: Dict[str, List[Dict[str, Any]]] = {}
    for rec in all_results:
        by_page.setdefault(rec["page"], []).append(rec)
    for page_png in sorted((RUN_DIR / "pages").glob("*.png")):
        per_page.append({"page": page_png.name, "results": by_page.get(page_png.name, [])})

    summary["per_page"] = per_page
    summary["checks"] = {
        "text_from_2_plus_models": any(
            sum(1 for r in p["results"] if "text_len" in r and r["text_len"] > 0) >= 2 for p in per_page
        ),
        "pages_processed": len(per_page),
    }
    (RUN_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n=== SUMMARY ===\n{json.dumps(summary, indent=2)}")

if __name__ == "__main__":
    main()
