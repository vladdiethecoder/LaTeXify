from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

try:  # pragma: no cover - optional dependency
    import yaml
except Exception:  # pragma: no cover - PyYAML optional
    yaml = None

from pypdf import PdfReader

from latexify.ingestion.backends import (
    FlorenceBackend,
    InternVLBackend,
    MinerUBackend,
    NougatBackend,
)
from latexify.ingestion.backends.base import BaseCouncilBackend, CouncilOutput, LayoutChunk
from latexify.ingestion.chunkers import fixed_chunk_pages, semantic_chunk_pages
from latexify.ingestion.orchestrator import IngestionPipeline
from latexify.utils.logging import configure_logging, log_info, log_warning

DEFAULT_ASSETS_DIR = Path("build/assets")
CHUNK_STRATEGIES = {
    "semantic": semantic_chunk_pages,
    "fixed": fixed_chunk_pages,
}
_INTERNVL_CONFIG_CACHE: Dict[str, Any] | None = None


def _load_internvl_config() -> Dict[str, Any]:
    global _INTERNVL_CONFIG_CACHE
    if _INTERNVL_CONFIG_CACHE is not None:
        return _INTERNVL_CONFIG_CACHE
    candidates: List[Path] = []
    env_path = os.environ.get("INTERNVL_CONFIG")
    if env_path:
        candidates.append(Path(env_path))
    default_path = Path("configs/ingestion/internvl.yaml")
    if not env_path:
        candidates.append(default_path)
    if yaml is None:
        _INTERNVL_CONFIG_CACHE = {}
        return _INTERNVL_CONFIG_CACHE
    for path in candidates:
        if not path or not path.exists():
            continue
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            data.setdefault("__config_path__", str(path))
            _INTERNVL_CONFIG_CACHE = data
            return _INTERNVL_CONFIG_CACHE
        except Exception as exc:
            log_warning("Failed to parse InternVL config", config=str(path), error=str(exc))
    _INTERNVL_CONFIG_CACHE = {}
    return _INTERNVL_CONFIG_CACHE


def _parse_max_memory_string(payload: str) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for chunk in payload.split(","):
        chunk = chunk.strip()
        if not chunk or "=" not in chunk:
            continue
        device, value = chunk.split("=", 1)
        device = device.strip()
        value = value.strip()
        if device and value:
            result[device] = value
    return result


def _normalize_max_memory(value: Any) -> Dict[str, str]:
    if not value:
        return {}
    if isinstance(value, dict):
        return {str(k): str(v) for k, v in value.items()}
    if isinstance(value, str):
        return _parse_max_memory_string(value)
    if isinstance(value, list):
        combined: Dict[str, str] = {}
        for item in value:
            if isinstance(item, str):
                combined.update(_parse_max_memory_string(item))
            elif isinstance(item, dict):
                combined.update({str(k): str(v) for k, v in item.items()})
        return combined
    return {}


def _resolve_internvl_mode(candidate: str | None = None, config_value: str | None = None) -> str:
    mode = candidate or config_value or os.environ.get("INTERNVL_MODE") or "vllm"
    mode = mode.strip().lower()
    if mode not in {"vllm", "hf"}:
        return "vllm"
    return mode


def _hf_runner_options(config_section: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    config_data = dict(config_section or {})
    extra_args_cfg = config_data.get("extra_args") or []
    if isinstance(extra_args_cfg, list):
        extra_args_list = [str(item) for item in extra_args_cfg]
    else:
        extra_args_list = shlex.split(str(extra_args_cfg))
    extra_env_cfg = config_data.get("env") or {}
    options: Dict[str, Any] = {
        "runner_script": config_data.get("script"),
        "model_dir": config_data.get("model_dir"),
        "python": config_data.get("python"),
        "max_new_tokens": config_data.get("max_new_tokens", 700),
        "extra_args": extra_args_list,
        "device_map": config_data.get("device_map"),
        "max_memory": _normalize_max_memory(config_data.get("max_memory")),
        "cuda_visible_devices": config_data.get("gpus") or config_data.get("cuda_visible_devices"),
        "offload_folder": config_data.get("offload_folder"),
        "timeout_seconds": config_data.get("timeout_seconds"),
        "env_overrides": {str(k): str(v) for k, v in (extra_env_cfg.items() if isinstance(extra_env_cfg, dict) else [])},
        "retries": config_data.get("retries", 1),
    }
    try:
        options["max_new_tokens"] = int(options["max_new_tokens"])
    except (TypeError, ValueError):
        options["max_new_tokens"] = 700
    if options["timeout_seconds"] is not None:
        try:
            options["timeout_seconds"] = int(options["timeout_seconds"])
        except (TypeError, ValueError):
            options["timeout_seconds"] = None
    extra_args_env = os.environ.get("INTERNVL_HF_EXTRA_ARGS", "")
    if extra_args_env:
        options["extra_args"].extend(shlex.split(extra_args_env))
    raw_max = os.environ.get("INTERNVL_HF_MAX_NEW_TOKENS")
    try:
        if raw_max is not None:
            options["max_new_tokens"] = int(raw_max)
    except ValueError:
        log_warning("Invalid INTERNVL_HF_MAX_NEW_TOKENS value", value=raw_max)
    options["runner_script"] = os.environ.get("INTERNVL_HF_RUNNER") or options["runner_script"]
    options["model_dir"] = os.environ.get("INTERNVL_HF_MODEL_DIR") or options["model_dir"]
    options["python"] = os.environ.get("INTERNVL_HF_PYTHON") or options["python"]
    options["device_map"] = os.environ.get("INTERNVL_HF_DEVICE_MAP") or options["device_map"]
    options["cuda_visible_devices"] = os.environ.get("INTERNVL_HF_GPUS") or os.environ.get("INTERNVL_HF_VISIBLE_GPUS") or options["cuda_visible_devices"]
    max_memory_env = os.environ.get("INTERNVL_HF_MAX_MEMORY")
    if max_memory_env:
        options["max_memory"].update(_parse_max_memory_string(max_memory_env))
    offload_env = os.environ.get("INTERNVL_HF_OFFLOAD_FOLDER")
    if offload_env:
        options["offload_folder"] = offload_env
    timeout_env = os.environ.get("INTERNVL_HF_TIMEOUT")
    if timeout_env:
        try:
            options["timeout_seconds"] = int(timeout_env)
        except ValueError:
            log_warning("Invalid INTERNVL_HF_TIMEOUT value", value=timeout_env)
    retries_env = os.environ.get("INTERNVL_HF_RETRIES")
    if retries_env:
        try:
            options["retries"] = max(1, int(retries_env))
        except ValueError:
            log_warning("Invalid INTERNVL_HF_RETRIES value", value=retries_env)
    options["retries"] = max(1, int(options["retries"] or 1))
    if not options["extra_args"]:
        options["extra_args"] = None
    if not options["max_memory"]:
        options["max_memory"] = None
    if not options["env_overrides"]:
        options["env_overrides"] = None
    return options


class GenericOCRBackend(BaseCouncilBackend):
    """Lightweight permissive fallback that reuses raw PDF text."""

    name = "generic_ocr"
    permissive = True

    async def process(self, chunk: LayoutChunk) -> CouncilOutput:
        text = (chunk.text or "").strip() or f"[permissive-fallback chunk={chunk.chunk_id}]"
        metadata = {
            "fallback_used": True,
            "reason": "permissive-backup",
            "source": self.name,
        }
        return CouncilOutput(
            backend=self.name,
            chunk_id=chunk.chunk_id,
            page_index=chunk.page_index,
            text=text,
            confidence=0.42,
            metadata=metadata,
        )


def _extract_pdf_text(pdf: Path, max_pages: int | None = None) -> List[str]:
    reader = PdfReader(str(pdf))
    pages: List[str] = []
    for idx, page in enumerate(reader.pages):
        if max_pages is not None and idx >= max_pages:
            break
        pages.append(page.extract_text() or "")
    return pages


def _sanitize_prefix(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_-]+", "_", value or "page")
    return sanitized or "page"


def _parse_pdfimages_page(filename: str, prefix: str) -> int | None:
    stem = Path(filename).stem
    marker = f"{prefix}-"
    if not stem.startswith(marker):
        return None
    remainder = stem[len(marker):]
    if "-" in remainder:
        candidate = remainder.split("-", 1)[0]
    else:
        candidate = remainder
    try:
        page_number = int(candidate)
    except ValueError:
        return None
    return max(page_number, 1) - 1


def _extract_with_pdfimages(pdf: Path, assets_dir: Path, max_pages: int | None = None) -> Mapping[int, List[Path]]:
    mapping: dict[int, List[Path]] = defaultdict(list)
    pdfimages = shutil.which("pdfimages")
    if not pdfimages:
        return mapping
    prefix = _sanitize_prefix(pdf.stem)
    with tempfile.TemporaryDirectory(prefix="latexify-pdfimages-") as tmp_dir:
        cmd = [pdfimages, "-p", "-all"]
        if max_pages is not None:
            cmd.extend(["-f", "1", "-l", str(max_pages)])
        cmd.extend([str(pdf), prefix])
        try:
            subprocess.run(
                cmd,
                cwd=tmp_dir,
                check=True,
                capture_output=True,
                text=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            stderr = ""
            stdout = ""
            if isinstance(exc, subprocess.CalledProcessError):
                stderr = exc.stderr or ""
                stdout = exc.stdout or ""
            log_warning(
                "pdfimages extraction failed",
                pdf=str(pdf),
                error=str(exc),
                stderr=stderr.strip(),
                stdout=stdout.strip(),
            )
            return {}
        temp_root = Path(tmp_dir)
        per_page_counts: dict[int, int] = defaultdict(int)
        for asset in sorted(temp_root.glob(f"{prefix}-*")):
            page_index = _parse_pdfimages_page(asset.name, prefix)
            if page_index is None:
                continue
            if max_pages is not None and page_index >= max_pages:
                continue
            per_page_counts[page_index] += 1
            suffix = asset.suffix or ".png"
            target = assets_dir / f"page{page_index+1:04d}-img{per_page_counts[page_index]:02d}{suffix}"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(asset), target)
            mapping[page_index].append(target)
        return mapping


def _extract_with_pypdf(pdf: Path, assets_dir: Path, max_pages: int | None = None) -> Mapping[int, List[Path]]:
    mapping: dict[int, List[Path]] = defaultdict(list)
    reader = PdfReader(str(pdf))
    for page_index, page in enumerate(reader.pages):
        if max_pages is not None and page_index >= max_pages:
            break
        page_images = getattr(page, "images", []) or []
        for img_idx, image in enumerate(page_images, start=1):
            name = getattr(image, "name", "") or f"page{page_index+1:04d}_img{img_idx:02d}.png"
            suffix = Path(name).suffix or ".png"
            out_path = assets_dir / f"page{page_index+1:04d}-img{img_idx:02d}{suffix}"
            try:
                out_path.write_bytes(image.data)  # type: ignore[attr-defined]
            except Exception as exc:  # pragma: no cover - corrupted streams are rare
                log_warning(
                    "Failed to persist extracted image",
                    page_index=page_index,
                    image=name,
                    error=str(exc),
                )
                continue
            mapping[page_index].append(out_path)
    return mapping


def _extract_page_images(pdf: Path, assets_dir: Path, max_pages: int | None = None) -> Mapping[int, List[Path]]:
    """Extract embedded images per page for downstream figure assembly."""

    mapping: dict[int, List[Path]] = defaultdict(list)
    if not pdf.exists():
        return mapping
    assets_dir.mkdir(parents=True, exist_ok=True)
    mapping = _extract_with_pdfimages(pdf, assets_dir, max_pages)
    if mapping:
        return mapping
    return _extract_with_pypdf(pdf, assets_dir, max_pages)


def _build_backends(
    *,
    internvl_endpoint: str | None,
    florence_endpoint: str | None,
    permissive: bool,
    internvl_mode: str | None = None,
) -> List[BaseCouncilBackend]:
    config = _load_internvl_config()
    config_mode = config.get("mode") if isinstance(config, dict) else None
    hf_section = config.get("hf_runner") if isinstance(config, dict) else None
    mode = _resolve_internvl_mode(internvl_mode, config_mode)
    hf_options = _hf_runner_options(hf_section) if mode == "hf" else None
    backends: List[BaseCouncilBackend] = [
        InternVLBackend(
            internvl_endpoint,
            mode=mode,
            hf_options=hf_options,
        ),
        FlorenceBackend(florence_endpoint),
        MinerUBackend(),
        NougatBackend(),
    ]
    if permissive:
        backends.append(GenericOCRBackend())
    return backends


async def ingest_pdf_async(
    pdf: Path,
    run_dir: Path,
    *,
    assets_dir: Path | None = None,
    max_pages: int | None = None,
    chunk_chars: int = 1100,
    chunk_strategy: str = "semantic",
    internvl_endpoint: str | None = None,
    florence_endpoint: str | None = None,
    pages_override: Sequence[str] | None = None,
    permissive: bool = False,
    internvl_mode: str | None = None,
) -> Dict[str, Any]:
    if chunk_strategy not in CHUNK_STRATEGIES:
        raise ValueError(f"Unknown chunk strategy {chunk_strategy}")
    if pages_override:
        pages = list(pages_override)
        images_by_page: Mapping[int, List[Path]] = {}
        image_root = assets_dir or DEFAULT_ASSETS_DIR
    else:
        if not pdf.exists():
            raise FileNotFoundError(pdf)
        pages = _extract_pdf_text(pdf, max_pages)
        image_root = assets_dir or DEFAULT_ASSETS_DIR
        images_by_page = _extract_page_images(pdf, image_root, max_pages=max_pages)
    run_dir.mkdir(parents=True, exist_ok=True)
    chunker = CHUNK_STRATEGIES[chunk_strategy]
    chunks = chunker(pages, chunk_chars=chunk_chars, images_by_page=images_by_page)

    backends = _build_backends(
        internvl_endpoint=internvl_endpoint,
        florence_endpoint=florence_endpoint,
        permissive=permissive,
        internvl_mode=internvl_mode,
    )
    pipeline = IngestionPipeline(
        pdf=pdf,
        run_dir=run_dir,
        chunks=chunks,
        backends=backends,
    )
    result = await pipeline.run_async()

    meta = {
        "source_pdf": str(pdf),
        "page_count": len(pages),
        "chunk_count": len(chunks),
        "assets_dir": str(image_root),
        "backends": result["backends"],
        "consensus_dir": result["consensus_dir"],
        "resilience_report": result["resilience_report"],
        "permissive_ocr": permissive,
        "chunk_strategy": chunk_strategy,
        "figures_extracted": sum(len(paths) for paths in images_by_page.values()),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def ingest_pdf(
    pdf: Path,
    run_dir: Path,
    **kwargs: Any,
) -> Dict[str, Any]:
    return asyncio.run(ingest_pdf_async(pdf, run_dir, **kwargs))


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Multi-model ingestion council for PDFs")
    ap.add_argument("--pdf", type=Path, required=True, help="Input PDF path")
    ap.add_argument("--run-dir", type=Path, required=True, help="Run directory for council outputs")
    ap.add_argument("--assets-dir", type=Path, default=DEFAULT_ASSETS_DIR, help="Directory for rendered assets")
    ap.add_argument("--max-pages", type=int, default=None, help="Limit number of pages to ingest")
    ap.add_argument("--chunk-chars", type=int, default=1100, help="Max characters per chunk before splitting")
    ap.add_argument(
        "--chunk-strategy",
        choices=sorted(CHUNK_STRATEGIES),
        default="semantic",
        help="Chunking algorithm to use",
    )
    ap.add_argument("--internvl-endpoint", type=str, default=None, help="OpenAI-compatible endpoint for InternVL")
    ap.add_argument(
        "--internvl-mode",
        choices=["vllm", "hf"],
        default=None,
        help="Select InternVL backend mode (default: environment or vllm).",
    )
    ap.add_argument("--florence-endpoint", type=str, default=None, help="OpenAI-compatible endpoint for Florence-2")
    ap.add_argument("--permissive", action="store_true", help="Enable generic OCR fallback backend")
    ap.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return ap


async def _run_cli(args: argparse.Namespace) -> int:
    configure_logging(verbose=args.verbose)
    log_info("Starting ingest_pdf", pdf=str(args.pdf), run_dir=str(args.run_dir))
    summary = await ingest_pdf_async(
        args.pdf,
        args.run_dir,
        assets_dir=args.assets_dir,
        max_pages=args.max_pages,
        chunk_chars=args.chunk_chars,
        chunk_strategy=args.chunk_strategy,
        internvl_endpoint=args.internvl_endpoint,
        internvl_mode=args.internvl_mode,
        florence_endpoint=args.florence_endpoint,
        permissive=args.permissive,
    )
    log_info("ingest_pdf completed", summary=summary)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = build_arg_parser()
    args = ap.parse_args(argv)
    return asyncio.run(_run_cli(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
