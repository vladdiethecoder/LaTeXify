from __future__ import annotations

import argparse
import logging
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from time import perf_counter
from pathlib import Path
from typing import Dict, Any, List

try:  # pragma: no cover - optional dependency
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

ROOT = Path(__file__).parent
BUILD_ROOT = ROOT / "build"
RELEASE_DIR = ROOT / "release"
INPUT_DIR = RELEASE_DIR / "inputs"
HF_CACHE_DIR = RELEASE_DIR / "models" / "hf_cache"
OUTPUT_DIR = BUILD_ROOT / "runs"
from release.core.model_paths import resolve_models_root  # noqa: E402

MODELS_DIR = resolve_models_root(ROOT / "models")
_HF_CACHE_READY = False

_ALLOC_CONF = "expandable_segments:True,max_split_size_mb:512"
if "PYTORCH_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_ALLOC_CONF"] = _ALLOC_CONF
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = _ALLOC_CONF


def _init_cache_root(base: Path) -> None:
    base.mkdir(parents=True, exist_ok=True)
    for subdir in ("hub", "transformers", "datasets"):
        (base / subdir).mkdir(parents=True, exist_ok=True)


def ensure_local_cache_dirs() -> Path:
    """Force Hugging Face caches into release/models/hf_cache if current env is unwritable."""

    global _HF_CACHE_READY
    if _HF_CACHE_READY and (hf_home := os.environ.get("HF_HOME")):
        return Path(hf_home)

    candidates = []
    if env_home := os.environ.get("HF_HOME"):
        candidates.append(Path(env_home))
    candidates.append(HF_CACHE_DIR)

    chosen: Path | None = None
    for candidate in candidates:
        try:
            _init_cache_root(candidate)
            chosen = candidate
            break
        except PermissionError:
            logging.warning("Cannot write to HF cache dir %s; trying fallback.", candidate)
            continue
    if chosen is None:
        _init_cache_root(HF_CACHE_DIR)
        chosen = HF_CACHE_DIR

    os.environ["HF_HOME"] = str(chosen)

    hub_dir = chosen / "hub"
    for env_name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        os.environ[env_name] = str(hub_dir)

    transformers_cache = chosen / "transformers"
    os.environ["TRANSFORMERS_CACHE"] = str(transformers_cache)

    datasets_cache = chosen / "datasets"
    os.environ["HF_DATASETS_CACHE"] = str(datasets_cache)

    _HF_CACHE_READY = True
    return chosen


# Configure Hugging Face cache paths before importing heavy deps that read env vars.
ensure_local_cache_dirs()

HALLUCINATION_MODEL_NAME = os.environ.get("LATEXIFY_HALLUCINATION_MODEL", "deepseek-ai/DeepSeek-V3")
VALIDATION_MODEL_NAME = os.environ.get("LATEXIFY_VALIDATION_MODEL", "Qwen/Qwen2.5-7B-Instruct")
VISUAL_MODEL_NAME = os.environ.get("LATEXIFY_VISUAL_JUDGE_MODEL", "Qwen/Qwen2.5-VL-32B-Instruct")


try:  # Optional dependency for memory checkpoints
    import psutil  # type: ignore
except Exception:  # pragma: no cover - psutil always available in release env
    psutil = None  # type: ignore


from release.pipeline import (  # noqa: E402  (import after cache bootstrap)
    active_learning,
    assembly,
    critique,
    ingestion,
    layout,
    metrics,
    planner,
    rag,
    reward,
    retrieval,
    structure_graph,
    synthesis,
    synthesis_coverage,
    validation,
    hallucination,
    visual_regression,
)
from release.pipeline.domain_detector import DomainDetector  # noqa: E402
from release.pipeline.semantic_chunking import SemanticChunker  # noqa: E402
from release.pipeline.quality_assessment import QualityAssessor  # noqa: E402
from release.pipeline.semantic_enricher import SemanticEnricher  # noqa: E402
from release.pipeline.iterative_refiner import IterativeRefiner  # noqa: E402
from release.pipeline.preamble_optimizer import optimize_preamble  # noqa: E402
from release.pipeline.bibliography import generate_bibliography  # noqa: E402
from release.pipeline.variable_consistency import normalize_variables  # noqa: E402
from release.pipeline.comment_generator import add_section_comments  # noqa: E402
from release.pipeline import reward_mm  # noqa: E402
from release.models import llm_refiner  # noqa: E402
from release.utils import quality  # noqa: E402
from release.core.config import BackendToggleConfig  # noqa: E402
from release.core.data_pathway_logger import init_logger  # noqa: E402


class StageLogger:
    """Append-only checkpoint log that survives terminal crashes."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def _gpu_snapshot(self) -> list[dict[str, float]] | None:
        if torch is None:
            return None
        try:
            if not torch.cuda.is_available():
                return None
        except Exception:
            return None
        stats: list[dict[str, float]] = []
        try:
            device_count = torch.cuda.device_count()
        except Exception:
            device_count = 0
        for idx in range(device_count):
            try:
                with torch.cuda.device(idx):
                    free_bytes, total_bytes = torch.cuda.mem_get_info()
            except Exception:
                continue
            used_bytes = max(0, total_bytes - free_bytes)
            stats.append(
                {
                    "device": f"cuda:{idx}",
                    "free_gb": round(free_bytes / 1024**3, 3),
                    "used_gb": round(used_bytes / 1024**3, 3),
                    "total_gb": round(total_bytes / 1024**3, 3),
                }
            )
        return stats or None

    def _mem_snapshot(self) -> Dict[str, Any] | None:
        if psutil is None:
            return None
        try:
            proc = psutil.Process(os.getpid())
            rss = proc.memory_info().rss / 1024**3
            vmem = psutil.virtual_memory()
            payload: Dict[str, Any] = {
                "proc_rss_gb": round(rss, 3),
                "system_used_gb": round(vmem.used / 1024**3, 3),
                "system_total_gb": round(vmem.total / 1024**3, 3),
            }
            cuda_stats = self._gpu_snapshot()
            if cuda_stats:
                payload["cuda"] = cuda_stats
            return payload
        except Exception:
            return None

    def log(self, stage: str, status: str, extra: Dict[str, Any] | None = None) -> None:
        payload: Dict[str, Any] = {
            "timestamp": self._timestamp(),
            "stage": stage,
            "status": status,
        }
        if extra:
            payload.update(extra)
        mem = self._mem_snapshot()
        if mem:
            payload["memory"] = mem
        try:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload) + "\n")
        except Exception:
            logging.debug("Failed to write checkpoint entry for %s", stage, exc_info=True)


class ProvenanceTracker:
    """Accumulates provenance metadata for JSON outputs."""

    def __init__(self) -> None:
        self._entries: Dict[str, Dict[str, Any]] = {}

    def _normalize(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(ROOT))
        except Exception:
            return str(path.resolve())

    def record(self, path: Path | None, stage: str, models: List[str] | None = None, notes: str | None = None) -> None:
        if path is None or path.suffix.lower() != ".json":
            return
        key = self._normalize(path)
        self._entries[key] = {
            "stage": stage,
            "models": models or [],
            "notes": notes or "",
        }

    def write(self, output_path: Path, run_id: str) -> None:
        if not self._entries:
            return
        payload = {
            "run_id": run_id,
            "artifacts": self._entries,
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _dir_has_payload(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def ensure_math_models_available(pdf_path: Path, models_root: Path) -> None:
    """Require local LayoutLM + pix2tex checkpoints for math-heavy PDFs."""

    if "math" not in pdf_path.stem.lower():
        return
    layout_dir = models_root / "layout" / "layoutlmv3-base"
    pix_dir = models_root / "ocr" / "pix2tex-base"
    missing = []
    if not _dir_has_payload(layout_dir):
        missing.append(
            (
                "LayoutLMv3-base",
                layout_dir,
                "layout/layoutlmv3-base",
            )
        )
    if not _dir_has_payload(pix_dir):
        missing.append(
            (
                "pix2tex-base",
                pix_dir,
                "ocr/pix2tex-base",
            )
        )
    if not missing:
        return
    lines = [
        f"- {name} (expected at {target}) → install via `python release/scripts/install_models.py --models {key}`"
        for name, target, key in missing
    ]
    raise RuntimeError(
        "Math-intensive PDF detected but required local models are missing:\n"
        + "\n".join(lines)
        + "\nThese checkpoints are now mandatory for math documents so that pix2tex + LayoutLM can run locally."
    )


def clean_release_root_artifacts() -> None:
    """Delete stray LaTeX artefacts accidentally written to release/."""

    patterns = ["main.*", "*.aux", "*.log", "*.fls", "*.fdb_latexmk", "*.pdf"]
    for base in (RELEASE_DIR, ROOT):
        for pattern in patterns:
            for path in base.glob(pattern):
                try:
                    path.unlink()
                except FileNotFoundError:
                    continue


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )


def run_chktex_lint(tex_path: Path, output_path: Path) -> Dict[str, Any]:
    """Run chktex linting if available and persist the results."""

    report: Dict[str, Any] = {
        "tool": "chktex",
        "available": False,
        "issues": [],
        "return_code": None,
    }
    binary = shutil.which("chktex")
    if not binary:
        report["notes"] = "chktex not installed"
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    report["available"] = True
    cmd = [binary, "-q", "-n1", "-I0", tex_path.name]
    try:
        proc = subprocess.run(
            cmd,
            cwd=tex_path.parent,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        report["notes"] = f"chktex invocation failed: {exc}"
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    raw_output = "\n".join(
        segment.strip() for segment in (proc.stdout, proc.stderr) if segment.strip()
    )
    issues = [line for line in raw_output.splitlines() if line.strip()]
    report["issues"] = issues
    report["return_code"] = proc.returncode
    report["notes"] = f"{len(issues)} warning(s)" if issues else "clean"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def resolve_pdf(path_str: str) -> Path:
    path = Path(path_str)
    if path.exists():
        return path
    candidate = INPUT_DIR / path_str
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"PDF not found: {path_str}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the simplified LaTeXify release pipeline.")
    parser.add_argument("--pdf", required=True, help="Path to the draft PDF (absolute or relative to release/inputs)")
    parser.add_argument("--title", default=None, help="Title for the generated LaTeX document")
    parser.add_argument("--author", default="LaTeXify Release", help="Author name for the document")
    parser.add_argument("--run-dir", default=None, help="Optional output directory for artifacts")
    parser.add_argument("--chunk-chars", type=int, default=ingestion.DEFAULT_CHUNK_CHARS, help="Maximum characters per chunk")
    parser.add_argument("--skip-compile", action="store_true", help="Skip PDF compilation (generate .tex only)")
    parser.add_argument(
        "--no-unicode-sanitizer",
        action="store_true",
        help="Disable Unicode→LaTeX normalization in the assembly stage (debug only).",
    )
    parser.add_argument("--log-level", default="INFO", help="Python logging level (INFO, DEBUG, ...)")
    parser.add_argument("--benchmark-dir", default=None, help="Optional directory of PDFs for READOC-style benchmarking")
    parser.add_argument("--benchmark-limit", type=int, default=5, help="Maximum PDFs to benchmark in one run")
    parser.add_argument(
        "--rag-cache",
        default=None,
        help="Path to persistent RAG cache (default: release/cache/rag_index.json)",
    )
    parser.add_argument(
        "--rag-cache-budget-mb",
        type=int,
        default=128,
        help="Approximate in-memory budget for exemplar cache (0 disables pruning).",
    )
    parser.add_argument("--rag-refresh", action="store_true", help="Force rebuilding the RAG cache before running")
    parser.add_argument(
        "--reward-mode",
        choices=["heuristic", "mm"],
        default="heuristic",
        help="Choose the aesthetic scorer used for rewards",
    )
    parser.add_argument(
        "--llm-mode",
        choices=["auto", "off"],
        default="auto",
        help="Enable or disable the LLM refinement stage",
    )
    parser.add_argument(
        "--llm-repo",
        default=None,
        help="Optional Hugging Face repo id for the LLM refiner (defaults to deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct)",
    )
    parser.add_argument(
        "--llm-backend",
        choices=["hf", "llama-cpp"],
        default=llm_refiner.DEFAULT_LLM_BACKEND,
        help="Backend used for the LLM refiner (transformers or llama.cpp).",
    )
    parser.add_argument(
        "--llama-cpp-model",
        default=llm_refiner.DEFAULT_LLAMA_MODEL,
        help="Path to a local codellama-13b-math GGUF checkpoint used when --llm-backend=llama-cpp.",
    )
    parser.add_argument(
        "--llama-cpp-grammar",
        default=llm_refiner.DEFAULT_LLAMA_GRAMMAR,
        help="Optional custom grammar file for the llama.cpp backend.",
    )
    parser.add_argument(
        "--style-domain",
        default="default",
        help="Domain under release/reference_tex used as curated style exemplars (use 'none' to disable).",
    )
    parser.add_argument(
        "--chunker-distance-threshold",
        type=float,
        default=None,
        help="Override semantic chunker distance threshold (default: 0.42).",
    )
    parser.add_argument(
        "--chunker-min-sentences",
        type=int,
        default=None,
        help="Minimum sentences per chunk before allowing a breakpoint.",
    )
    parser.add_argument(
        "--chunker-backend",
        choices=["auto", "hf", "hash"],
        default=None,
        help="Encoder backend for semantic chunking.",
    )
    parser.add_argument(
        "--chunker-encoder-name",
        default=None,
        help="Hugging Face encoder name used by the semantic chunker.",
    )
    parser.add_argument(
        "--chunker-download",
        choices=["auto", "always", "never"],
        default="auto",
        help="Control chunker model downloads (auto=use defaults, always=force allow, never=force offline).",
    )
    parser.add_argument(
        "--chunker-hash-fallback",
        choices=["auto", "allow", "disable"],
        default="auto",
        help="Control hashing fallback usage when encoders are unavailable.",
    )
    parser.add_argument(
        "--ocr-backend",
        choices=["florence", "ensemble", "mineru"],
        default="ensemble",
        help="Choose the primary ingestion backend (florence2-only, Florence+InternVL ensemble, or MinerU placeholder).",
    )
    parser.add_argument(
        "--math-ocr",
        choices=["none", "pix2tex", "latex-ocr"],
        default="none",
        help="Select the specialized math OCR backend used for equation crops.",
    )
    parser.add_argument(
        "--marker-backup",
        action="store_true",
        help="Enable Marker-backed fallback ingestion when the primary backend fails.",
    )
    parser.add_argument(
        "--mcp-pdf-processor",
        action="store_true",
        help="Enable the MCP PDF processor adapter (local MCP server required).",
    )
    return parser.parse_args()


def _configure_chunker(args: argparse.Namespace) -> SemanticChunker | None:
    overrides: Dict[str, object] = {}
    if args.chunker_distance_threshold is not None:
        overrides["distance_threshold"] = args.chunker_distance_threshold
    if args.chunker_min_sentences is not None:
        overrides["min_sentences_per_chunk"] = args.chunker_min_sentences
    if args.chunker_backend:
        overrides["encoder_backend"] = args.chunker_backend
    if args.chunker_encoder_name:
        overrides["encoder_name"] = args.chunker_encoder_name
    download_mode = args.chunker_download
    if download_mode == "always":
        overrides["allow_model_download"] = True
    elif download_mode == "never":
        overrides["allow_model_download"] = False
    hash_mode = args.chunker_hash_fallback
    if hash_mode == "allow":
        overrides["allow_hash_fallback"] = True
    elif hash_mode == "disable":
        overrides["allow_hash_fallback"] = False
    if not overrides:
        return None
    return SemanticChunker(**overrides)


def run_pipeline(args: argparse.Namespace) -> Path:
    ensure_local_cache_dirs()
    clean_release_root_artifacts()
    configure_logging(args.log_level)
    pdf_path = resolve_pdf(args.pdf)
    ensure_math_models_available(pdf_path, MODELS_DIR)
    timestamp_token = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    default_run_name = f"{pdf_path.stem}_{timestamp_token}"
    run_name = args.run_dir or default_run_name
    run_dir = Path(run_name)
    if not run_dir.is_absolute():
        if run_dir.parts and run_dir.parts[0] == "release":
            run_dir = ROOT / run_dir
        else:
            run_dir = OUTPUT_DIR / run_dir
    if run_dir.resolve() == RELEASE_DIR.resolve():
        logging.warning(
            "Run directory %s points to release/. Redirecting artifacts into build/runs instead.",
            run_dir,
        )
        fallback_name = f"{Path(run_name).stem}_rel"
        run_dir = OUTPUT_DIR / fallback_name
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = run_dir / "artifacts"
    reports_dir = run_dir / "reports"
    logs_dir = run_dir / "logs"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    agent_stats: Dict[str, Dict[str, float]] = {}
    stage_logger = StageLogger(logs_dir / "checkpoint.log")
    provenance = ProvenanceTracker()
    run_id_source = Path(run_name).name or default_run_name

    def _sanitize_run_id(value: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
        return safe or timestamp_token

    run_id = _sanitize_run_id(run_id_source)

    def _rel(path: Path) -> str:
        try:
            return str(path.relative_to(ROOT))
        except ValueError:
            return str(path)

    build_run_dir = BUILD_ROOT / f"run-{run_id}"
    build_run_dir.mkdir(parents=True, exist_ok=True)
    data_logger = init_logger(
        run_id,
        build_run_dir,
        {
            "input_pdf": _rel(pdf_path),
            "artifact_dir": _rel(run_dir),
            "args": {
                "chunk_chars": args.chunk_chars,
                "skip_compile": args.skip_compile,
                "reward_mode": args.reward_mode,
                "llm_mode": args.llm_mode,
            },
        },
    )

    def _short(text: str) -> str:
        if not text:
            return ""
        return text if len(text) <= 200 else text[:197] + "..."

    def log_stage_event(stage: str, status: str, **extra: Any) -> None:
        event = {"stage": stage, "status": status}
        event.update(extra)
        data_logger.log_event(event)

    def telemetry_stage(stage: str, status: str, **extra: Any) -> None:
        """Proxy used by deep pipeline components to emit structured events."""

        log_stage_event(stage, status, **extra)

    def run_agent(name, fn, *fn_args, **fn_kwargs):
        stage_logger.log(name, "start")
        start = perf_counter()
        try:
            result = fn(*fn_args, **fn_kwargs)
        except Exception as exc:
            stage_logger.log(name, "error", {"error": repr(exc)})
            raise
        duration = round(perf_counter() - start, 3)
        agent_stats[name] = {"duration_sec": duration}
        stage_logger.log(name, "success", {"duration_sec": duration})
        return result

    llm_refiner_instance = None
    llm_refiner_config: llm_refiner.LLMRefinerConfig | None = None
    if args.llm_mode != "off":
        llm_refiner_config = llm_refiner.LLMRefinerConfig(
            repo_id=args.llm_repo or llm_refiner.DEFAULT_LLM_REPO,
            style_domain=args.style_domain,
            backend=args.llm_backend,
            llama_cpp_model=args.llama_cpp_model,
            llama_grammar_path=args.llama_cpp_grammar,
        )
    else:
        logging.info("[refiner] LLM mode disabled via --llm-mode=off; template specialists only.")

    def ensure_llm_refiner() -> None:
        nonlocal llm_refiner_instance, llm_refiner_config
        if llm_refiner_instance is not None or llm_refiner_config is None:
            return
        stage_logger.log("LLMRefiner", "start")
        log_stage_event("llm_refiner", "started", models=[llm_refiner_config.repo_id], notes="Loading LLM refiner.")
        try:
            llm_refiner_instance = llm_refiner.LLMRefiner(llm_refiner_config)
            stage_logger.log("LLMRefiner", "success")
            active_repo = getattr(llm_refiner_instance, "active_repo_id", llm_refiner_config.repo_id)
            log_stage_event("llm_refiner", "completed", models=[active_repo])
        except llm_refiner.LLMRefinerError as exc:  # pragma: no cover - heavy dependency
            logging.warning("[refiner] Falling back to template specialists: %s", exc)
            llm_refiner_config = None
            stage_logger.log("LLMRefiner", "error", {"error": str(exc)})
            log_stage_event("llm_refiner", "failed", notes=_short(str(exc)))
        except Exception as exc:  # pragma: no cover - heavy dependency
            logging.exception("[refiner] Unexpected error while loading LLM refiner, falling back")
            llm_refiner_config = None
            stage_logger.log("LLMRefiner", "error", {"error": repr(exc)})
            log_stage_event("llm_refiner", "failed", notes=_short(repr(exc)))

    run_start = perf_counter()
    stage_logger.log("Run", "start", {"pdf": str(pdf_path)})
    log_stage_event(
        "run",
        "started",
        input_files=[_rel(pdf_path)],
        notes="Pipeline initialized.",
    )
    backend_config = BackendToggleConfig(
        ocr_backend=args.ocr_backend,
        mineru_enabled=args.ocr_backend == "mineru",
        marker_enabled=args.marker_backup,
        mcp_pdf_processor_enabled=args.mcp_pdf_processor,
        math_ocr_backend=args.math_ocr,
    )
    ingestion_mode = backend_config.resolve_ingestion_mode()
    if backend_config.ocr_backend == "mineru" and ingestion_mode != "mineru":
        logging.info(
            "MinerU backend requested; adapter not yet wired so using %s until MinerU integration lands.",
            ingestion_mode,
        )
    if backend_config.marker_enabled:
        logging.info("Marker fallback flag enabled (adapter scaffolding only for now).")
    if backend_config.mcp_pdf_processor_enabled:
        logging.info("MCP PDF processor toggle enabled; awaiting server adapter.")
    if backend_config.wants_math_ocr():
        logging.info("Math OCR backend requested: %s", backend_config.math_ocr_backend)
    chunker = _configure_chunker(args)
    try:
        ingestion_models = [
            "nougat-small",
            "florence2-large",
            f"internvl:{ingestion.INTERNVL_MODEL_ID}",
            "trocr-math",
            "pix2tex-base",
        ]
        log_stage_event(
            "ingestion",
            "started",
            input_files=[_rel(pdf_path)],
            models=ingestion_models,
            notes=f"chunk_chars={args.chunk_chars}, backend={backend_config.ocr_backend}/{ingestion_mode}",
        )
        try:
            chunks_result = run_agent(
                "ParsingAgent",
                ingestion.run_ingestion,
                pdf_path=pdf_path,
                workspace=artifacts_dir,
                chunk_chars=args.chunk_chars,
                ocr_mode=ingestion_mode,
                capture_page_images=True,
                models_dir=MODELS_DIR,
                semantic_chunker=chunker,
                telemetry=telemetry_stage,
                backend_config=backend_config,
            )
        except Exception as exc:
            log_stage_event("ingestion", "failed", notes=_short(repr(exc)))
            raise
        log_stage_event(
            "ingestion",
            "completed",
            output_files=[
                _rel(chunks_result.chunks_path),
                _rel(artifacts_dir),
            ],
            notes="Workspace artifacts ready.",
        )
        provenance.record(chunks_result.chunks_path, "ingestion", ingestion_models, "chunked text")
        provenance.record(chunks_result.tree_path, "ingestion", ingestion_models, "document tree")
        provenance.record(chunks_result.document_path, "ingestion", ingestion_models, "document representation")
        provenance.record(chunks_result.manifest_path, "ingestion", ingestion_models, "ingestion manifest")
        input_quality_profile: Dict[str, object] = chunks_result.quality_profile or {}
        input_quality_path = reports_dir / "input_quality.json"
        input_quality_path.write_text(json.dumps(input_quality_profile, indent=2), encoding="utf-8")
        log_stage_event(
            "input_quality",
            "completed",
            output_files=[_rel(input_quality_path)],
            notes=f"tier={input_quality_profile.get('tier', 'unknown')}",
        )
        provenance.record(input_quality_path, "input_quality", [], "ingestion quality profile")
        quality_profile = input_quality_profile
        quality_mode = str(quality_profile.get("processing_mode", "balanced"))
        master_plan_path = reports_dir / "master_plan.json"
        title = args.title or pdf_path.stem
        log_stage_event(
            "planning",
            "started",
            input_files=[_rel(chunks_result.chunks_path)],
            notes="Building master plan.",
        )
        try:
            run_agent(
                "PlannerAgent",
                planner.run_planner,
                chunks_result.chunks_path,
                master_plan_path,
                title,
            )
        except Exception as exc:
            log_stage_event("planning", "failed", notes=_short(repr(exc)))
            raise
        log_stage_event(
            "planning",
            "completed",
            output_files=[_rel(master_plan_path)],
        )
        provenance.record(master_plan_path, "planning", [], "master plan")
        master_plan = planner.load_master_plan(master_plan_path)
        rag_source_dir = ROOT / "reference_tex"
        rag_cache_path = Path(args.rag_cache) if args.rag_cache else ROOT / "cache" / "rag_index.json"
        if rag_cache_path and not rag_cache_path.is_absolute():
            rag_cache_path = ROOT / rag_cache_path
        rag_cache_path.parent.mkdir(parents=True, exist_ok=True)
        if args.rag_refresh:
            rag.build_index(rag_source_dir, rag_cache_path)
        rag_index = rag.load_or_build_index(rag_source_dir, rag_cache_path)
        rag_budget_mb = max(0, args.rag_cache_budget_mb)
        if rag_budget_mb:
            rag_index = rag.optimize_index(rag_index, budget_mb=rag_budget_mb)
        retrieval_notes = f"cache={_rel(rag_cache_path)}"
        if rag_budget_mb:
            retrieval_notes = f"{retrieval_notes};budget={rag_budget_mb}MB"
        plan_path = reports_dir / "plan.json"
        log_stage_event(
            "layout",
            "started",
            input_files=[_rel(chunks_result.chunks_path)],
            notes="Creating structured plan.",
        )
        try:
            run_agent("LayoutAgent", layout.run_layout, chunks_result.chunks_path, plan_path, master_plan_path)
        except Exception as exc:
            log_stage_event("layout", "failed", notes=_short(repr(exc)))
            raise
        log_stage_event(
            "layout",
            "completed",
            output_files=[_rel(plan_path)],
        )
        provenance.record(plan_path, "layout", [], "layout plan")
        domain_detector = DomainDetector()
        domain_profile_data: Dict[str, object] = {}
        domain_profile_path = reports_dir / "domain_profile.json"
        log_stage_event(
            "domain_detection",
            "started",
            input_files=[_rel(chunks_result.chunks_path)],
        )
        try:
            domain_profile = domain_detector.analyze(chunks_result.chunks_path, plan_path)
            domain_profile_data = domain_profile.to_dict()
            domain_profile_path.write_text(json.dumps(domain_profile_data, indent=2), encoding="utf-8")
        except Exception as exc:
            log_stage_event("domain_detection", "failed", notes=_short(repr(exc)))
        else:
            log_stage_event(
                "domain_detection",
                "completed",
                output_files=[_rel(domain_profile_path)],
                notes=f"domain={domain_profile_data.get('domain', 'general')}",
            )
            provenance.record(domain_profile_path, "domain_detection", [], "domain profile")
        graph_path = reports_dir / "graph.json"
        log_stage_event(
            "structure_graph",
            "started",
            input_files=[_rel(plan_path)],
        )
        try:
            run_agent("StructureGraphAgent", structure_graph.build_graph, plan_path, chunks_result.chunks_path, graph_path)
        except Exception as exc:
            log_stage_event("structure_graph", "failed", notes=_short(repr(exc)))
            raise
        log_stage_event(
            "structure_graph",
            "completed",
            output_files=[_rel(graph_path)],
        )
        provenance.record(graph_path, "structure_graph", [], "structure graph")
        retrieval_path = reports_dir / "retrieval.json"
        log_stage_event(
            "retrieval",
            "started",
            input_files=[_rel(plan_path)],
            notes=retrieval_notes,
        )
        try:
            run_agent("RetrievalAgent", retrieval.build_index, chunks_result.chunks_path, plan_path, retrieval_path)
        except Exception as exc:
            log_stage_event("retrieval", "failed", notes=_short(repr(exc)))
            raise
        log_stage_event(
            "retrieval",
            "completed",
            output_files=[_rel(retrieval_path)],
        )
        provenance.record(retrieval_path, "retrieval", [], "retrieval index")
        snippets_path = reports_dir / "snippets.json"
        preamble_path = reports_dir / "preamble.json"
        ensure_llm_refiner()
        log_stage_event(
            "synthesis",
            "started",
            input_files=[_rel(plan_path), _rel(retrieval_path)],
            models=[llm_refiner_config.repo_id] if llm_refiner_config else [],
            notes=f"style={args.style_domain}",
        )
        try:
            run_agent(
                "LaTeXSynthAgent",
                synthesis.run_synthesis,
                chunks_result.chunks_path,
                plan_path,
                graph_path,
                retrieval_path,
                snippets_path,
                preamble_path,
                master_plan.document_class,
                master_plan.class_options,
                rag_index,
                llm_refiner=llm_refiner_instance,
                master_plan_path=master_plan_path,
                quality_profile=quality_profile,
                domain_profile=domain_profile_data,
            )
        except Exception as exc:
            log_stage_event("synthesis", "failed", notes=_short(repr(exc)))
            raise
        log_stage_event(
            "synthesis",
            "completed",
            output_files=[_rel(snippets_path)],
        )
        synthesis_models: List[str] = []
        if llm_refiner_instance and getattr(llm_refiner_instance, "active_repo_id", None):
            synthesis_models = [llm_refiner_instance.active_repo_id]
        elif llm_refiner_config:
            synthesis_models = [llm_refiner_config.repo_id]
        if preamble_path.exists():
            provenance.record(
                preamble_path,
                "synthesis",
                synthesis_models,
                f"generated preamble (domain={domain_profile_data.get('domain', 'general')})",
            )
        enrichment_report_path = reports_dir / "semantic_enrichment.json"
        log_stage_event(
            "semantic_enrichment",
            "started",
            input_files=[_rel(snippets_path)],
        )
        recorded_snippets = False
        try:
            enrichment_report = SemanticEnricher(domain_profile_data, quality_profile).enrich(
                plan_path,
                chunks_result.chunks_path,
                snippets_path,
            )
            enrichment_data = enrichment_report.to_dict()
            enrichment_report_path.write_text(json.dumps(enrichment_data, indent=2), encoding="utf-8")
        except Exception as exc:
            log_stage_event("semantic_enrichment", "failed", notes=_short(repr(exc)))
        else:
            updated = enrichment_data.get("updated", 0)
            log_stage_event(
                "semantic_enrichment",
                "completed",
                output_files=[_rel(enrichment_report_path)],
                notes=f"updated={updated}",
            )
            provenance.record(
                snippets_path,
                "semantic_enrichment",
                synthesis_models,
                f"snippets enriched (domain={domain_profile_data.get('domain', 'general')})",
            )
            recorded_snippets = True
        if not recorded_snippets:
            provenance.record(
                snippets_path,
                "synthesis",
                synthesis_models,
                "generated snippets",
            )
        use_sanitizer = not getattr(args, "no_unicode_sanitizer", False)

        log_stage_event(
            "assembly",
            "started",
            input_files=[_rel(plan_path), _rel(snippets_path)],
            notes="Building LaTeX and assets.",
        )
        try:
            tex_path = run_agent(
                "AssemblyAgent",
                assembly.run_assembly,
                plan_path=plan_path,
                snippets_path=snippets_path,
                preamble_path=preamble_path,
                output_dir=run_dir,
                title=title,
                author=args.author,
                skip_compile=args.skip_compile,
                use_unicode_sanitizer=use_sanitizer,
                chunks_path=chunks_result.chunks_path,
                domain_profile=domain_profile_data,
            )
        except Exception as exc:
            log_stage_event("assembly", "failed", notes=_short(repr(exc)))
            raise
        pdf_candidate = tex_path.with_suffix(".pdf")
        outputs = [_rel(tex_path)]
        if not args.skip_compile and pdf_candidate.exists():
            outputs.append(_rel(pdf_candidate))
        log_stage_event(
            "assembly",
            "completed",
            output_files=outputs,
        )
        log_stage_event(
            "preamble_opt",
            "started",
            input_files=[_rel(tex_path)],
        )
        try:
            changed = optimize_preamble(tex_path)
        except Exception as exc:
            log_stage_event("preamble_opt", "failed", notes=_short(repr(exc)))
        else:
            note = "optimized" if changed else "no-op"
        log_stage_event(
            "preamble_opt",
            "completed",
            notes=note,
        )
        bib_path = reports_dir / "latexify_auto.bib"
        log_stage_event("autobib", "started", input_files=[_rel(tex_path)])
        try:
            bib_report = generate_bibliography(tex_path, bib_path)
        except Exception as exc:
            log_stage_event("autobib", "failed", notes=_short(repr(exc)))
            bib_report = {"created": False}
        else:
            status = "completed" if bib_report.get("created") else "skipped"
            notes = f"entries={bib_report.get('count', 0)}"
            outputs_list = [_rel(bib_path)] if bib_report.get("created") else []
            log_stage_event("autobib", status, output_files=outputs_list, notes=notes)
            if bib_report.get("created"):
                provenance.record(bib_path, "autobib", [], "auto bibliography")
        log_stage_event("variable_consistency", "started")
        try:
            variable_report = normalize_variables(tex_path)
        except Exception as exc:
            log_stage_event("variable_consistency", "failed", notes=_short(repr(exc)))
        else:
            note = "updated" if variable_report.get("changed") else "no-op"
            log_stage_event("variable_consistency", "completed", notes=note)
            agent_stats["VariableConsistency"] = variable_report
        log_stage_event("comment_generation", "started")
        try:
            comment_report = add_section_comments(tex_path)
        except Exception as exc:
            log_stage_event("comment_generation", "failed", notes=_short(repr(exc)))
        else:
            log_stage_event(
                "comment_generation",
                "completed",
                notes=f"comments={comment_report.get('added', 0)}",
            )
        lint_path = reports_dir / "lint_chktex.json"
        log_stage_event(
            "lint",
            "started",
            input_files=[_rel(tex_path)],
            notes="chktex preflight",
        )
        lint_report = run_chktex_lint(tex_path, lint_path)
        lint_notes = lint_report.get("notes", "")
        if lint_report.get("issues"):
            sample = "; ".join(lint_report["issues"][:2])
            lint_notes = _short(sample) if sample else lint_notes
        log_stage_event(
            "lint",
            "completed",
            output_files=[_rel(lint_path)],
            notes=lint_notes,
        )
        provenance.record(lint_path, "lint", [], "chktex lint")
        quality_assessor = QualityAssessor()
        quality_path = reports_dir / "quality_report.json"
        log_stage_event(
            "quality_assessment",
            "started",
            input_files=[_rel(tex_path)],
        )
        try:
            quality_report = run_agent(
                "QualityAssessor",
                quality_assessor.evaluate,
                tex_path,
                chunks_result.chunks_path,
                plan_path,
                snippets_path,
            )
        except Exception as exc:
            log_stage_event("quality_assessment", "failed", notes=_short(repr(exc)))
            raise
        quality_path = reports_dir / "quality_report.json"
        quality_path.write_text(json.dumps(quality_report, indent=2), encoding="utf-8")
        log_stage_event(
            "quality_assessment",
            "completed",
            output_files=[_rel(quality_path)],
        )
        if quality_report.get("aggregate", 0.0) < quality_assessor.target_score:
            refiner = IterativeRefiner(
                assessor=quality_assessor,
                plan_path=plan_path,
                snippets_path=snippets_path,
                chunks_path=chunks_result.chunks_path,
                preamble_path=preamble_path,
                output_dir=run_dir,
                title=title,
                author=args.author,
                sanitize_unicode=use_sanitizer,
                skip_compile=args.skip_compile,
            )
            log_stage_event("iterative_refiner", "started")
            try:
                refinement = run_agent("IterativeRefiner", refiner.refine)
            except Exception as exc:
                log_stage_event("iterative_refiner", "failed", notes=_short(repr(exc)))
                raise
            tex_path = refinement.tex_path
            quality_report = refinement.report
            quality_path.write_text(json.dumps(quality_report, indent=2), encoding="utf-8")
            log_stage_event(
                "iterative_refiner",
                "completed",
                output_files=[_rel(tex_path)],
                notes="Applied repair loop.",
            )
        agent_stats["QualityScore"] = {"aggregate": quality_report.get("aggregate", 0.0)}
        provenance.record(quality_path, "quality_assessment", [], "quality report")
        issues = critique.evaluate_output(plan_path, tex_path)
        log_stage_event(
            "critique",
            "completed",
            notes="Issues detected" if issues else "No structural issues.",
        )
        if issues:
            logging.warning("Self-critique detected issues: %s", issues)
            fix = critique.attempt_fix(plan_path, retrieval_path, chunks_result.chunks_path, snippets_path)
            if fix:
                logging.info("Applied fix: %s", fix)
                ensure_llm_refiner()
                synthesis.run_synthesis(
                    chunks_result.chunks_path,
                    plan_path,
                    graph_path,
                    retrieval_path,
                    snippets_path,
                    preamble_path,
                    master_plan.document_class,
                    master_plan.class_options,
                    rag_index,
                    llm_refiner=llm_refiner_instance,
                    master_plan_path=master_plan_path,
                )
                tex_path = run_agent(
                    "AssemblyAgent",
                    assembly.run_assembly,
                    plan_path=plan_path,
                    snippets_path=snippets_path,
                    preamble_path=preamble_path,
                    output_dir=run_dir,
                    title=title,
                    author=args.author,
                    skip_compile=args.skip_compile,
                    use_unicode_sanitizer=use_sanitizer,
                    chunks_path=chunks_result.chunks_path,
                )
                refinement_report_path = run_dir / "refinement_report.json"
                if refinement_report_path.exists():
                    refinement_data = json.loads(refinement_report_path.read_text(encoding="utf-8"))
                    agent_stats["RefinementAgent"] = {
                        "passes": len(refinement_data.get("passes", [])),
                        "success": refinement_data.get("success", False),
                    }
                    provenance.record(refinement_report_path, "RefinementAgent", [], "compilation multi-pass report")
                citations_path = run_dir / "citations.json"
                if citations_path.exists():
                    citation_data = json.loads(citations_path.read_text(encoding="utf-8"))
                    agent_stats["CitationDetector"] = {
                        "count": len(citation_data.get("citations", [])),
                        "style": citation_data.get("dominant_style"),
                    }
                    provenance.record(citations_path, "CitationDetector", [], "citation detection report")
                issues = critique.evaluate_output(plan_path, tex_path)
        if issues:
            logging.warning("Remaining issues after fixes: %s", issues)
        else:
            logging.info("Self-critique passed")
        hallucination_path = reports_dir / "hallucination.json"
        hallucination_report: Dict[str, object] | None = None
        log_stage_event(
            "hallucination_check",
            "started",
            input_files=[_rel(plan_path)],
        )
        try:
            hallucination_report = hallucination.check_section_headers(
                plan_path,
                chunks_result.chunks_path,
                hallucination_path,
                snippets_path,
            )
        except Exception as exc:
            log_stage_event("hallucination_check", "failed", notes=_short(repr(exc)))
        else:
            flagged = hallucination_report.get("flagged_count", 0)
            notes = f"flagged={flagged}" if flagged else "clean"
            log_stage_event(
                "hallucination_check",
                "completed",
                output_files=[_rel(hallucination_path)],
                notes=notes,
            )
            provenance.record(hallucination_path, "hallucination_check", [], "header validation")
            agent_stats["HallucinationCheck"] = {"flagged": flagged}

        gaps_report = synthesis_coverage.find_gaps(master_plan_path, snippets_path)
        gaps_path = reports_dir / "synthesis_gaps.json"
        gaps_path.write_text(json.dumps(gaps_report, indent=2), encoding="utf-8")
        provenance.record(gaps_path, "synthesis_coverage", [], "coverage report")
        missing_chunks = gaps_report.get("missing_chunk_ids") or []
        missing_count = len(missing_chunks)
        agent_stats["SynthesisCoverage"] = {
            "expected": gaps_report.get("expected_snippets", 0),
            "actual": gaps_report.get("actual_snippets", 0),
            "missing_count": missing_count,
        }
        if missing_chunks:
            sample = missing_chunks[:10]
            agent_stats["SynthesisCoverage"]["sample_missing"] = sample
            logging.error(
                "Missing snippets for %s plan items (showing up to 10): %s",
                missing_count,
                sample,
            )
        validation_path = reports_dir / "validation.json"
        log_stage_event(
            "validation",
            "started",
            input_files=[_rel(tex_path)],
            notes="tectonic/latexmk auto",
        )
        try:
            validation_result = run_agent("ValidationAgent", validation.run_validation, tex_path, validation_path)
        except Exception as exc:
            log_stage_event("validation", "failed", notes=_short(repr(exc)))
            raise
        validation_errors = validation_result.get("errors") or []
        validation_notes = _short("; ".join(validation_errors)) if validation_errors else ""
        validation_status = "completed" if not validation_errors else "failed"
        log_stage_event(
            "validation",
            validation_status,
            output_files=[_rel(validation_path)],
            notes=validation_notes,
        )
        provenance.record(validation_path, "validation", [], "compile validation")
        if validation_errors:
            logging.warning("Validation errors detected: %s", validation_errors)
        metrics_path = reports_dir / "metrics.json"
        log_stage_event("metrics", "started")
        try:
            metrics.evaluate(
                plan_path,
                tex_path,
                retrieval_path,
                metrics_path,
                chunks_result.chunks_path,
                validation_path,
            )
        except Exception as exc:
            log_stage_event("metrics", "failed", notes=_short(repr(exc)))
            raise
        logging.info("Stage metrics written to %s", metrics_path)
        log_stage_event("metrics", "completed", output_files=[_rel(metrics_path)])
        provenance.record(metrics_path, "metrics", [], "stage metrics")
        consistency_report_path = reports_dir / "consistency.json"
        try:
            consistency_payload = reward_mm.visual_textual_consistency(
                tex_path,
                plan_path,
                chunks_result.chunks_path,
                output_path=consistency_report_path,
                assets_dir=run_dir / "assets",
            )
        except Exception as exc:
            logging.warning("Consistency verifier failed: %s", exc)
        else:
            agent_stats["VisualConsistencyVerifier"] = {
                "figure_checks": len(consistency_payload.get("figures", [])),
                "equation_checks": len(consistency_payload.get("equations", [])),
                "layout_score": consistency_payload.get("layout", {}).get("score", 0.0),
            }
            provenance.record(
                consistency_report_path,
                "VisualConsistencyVerifier",
                [],
                "visual-textual consistency report",
            )
        reward_path = reports_dir / "rewards.json"
        reward_models: List[str] = []
        if args.reward_mode == "mm":
            reward_models = [f"internvl:{os.environ.get('LATEXIFY_INTERNVL_MODEL', 'OpenGVLab/InternVL3_5-8B')}"]
        log_stage_event(
            "reward",
            "started",
            notes=f"mode={args.reward_mode}",
            models=reward_models,
        )
        try:
            reward.evaluate_rewards(
                chunks_result.chunks_path,
                tex_path,
                validation_path,
                reward_path,
                mode=args.reward_mode,
                trace_path=reports_dir / "reward_trace.jsonl",
            )
        except Exception as exc:
            log_stage_event("reward", "failed", notes=_short(repr(exc)))
            raise
        logging.info("Reward metrics written to %s", reward_path)
        log_stage_event(
            "reward",
            "completed",
            output_files=[_rel(reward_path)],
        )
        provenance.record(reward_path, "reward", reward_models, f"reward_{args.reward_mode}")
        reward_report: Dict[str, object] = {}
        if reward_path.exists():
            try:
                reward_report = json.loads(reward_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                reward_report = {}
        visual_report = None
        visual_stage_needed = not args.skip_compile and chunks_result.page_images_dir is not None
        if visual_stage_needed and not chunks_result.page_images_dir.exists():
            visual_stage_needed = False
            log_stage_event(
                "visual_regression",
                "skipped",
                notes="reference page renders missing",
            )
        if not args.skip_compile and chunks_result.page_images_dir is None:
            log_stage_event(
                "visual_regression",
                "skipped",
                notes="page image cache disabled",
            )
        elif args.skip_compile:
            log_stage_event("visual_regression", "skipped", notes="skip-compile enabled")
        if visual_stage_needed and not args.skip_compile and chunks_result.page_images_dir:
            visual_path = reports_dir / "visual_regression.json"
            log_stage_event(
                "visual_regression",
                "started",
                input_files=[_rel(tex_path)],
                notes="pdf2image diff",
            )
            try:
                visual_report = visual_regression.run_visual_regression(
                    tex_path,
                    chunks_result.page_images_dir,
                    visual_path,
                )
            except Exception as exc:
                log_stage_event("visual_regression", "failed", notes=_short(repr(exc)))
            else:
                status = "completed" if visual_report.get("available") else "skipped"
                flagged = visual_report.get("flagged_pages", 0)
                if flagged:
                    notes = f"flagged={flagged}"
                else:
                    notes = visual_report.get("reason", "clean")
                log_stage_event(
                    "visual_regression",
                    status,
                    output_files=[_rel(visual_path)],
                    notes=_short(notes) if notes else "",
                )
                if visual_report.get("available"):
                    provenance.record(visual_path, "visual_regression", [], "page diff")
                    agent_stats["VisualRegression"] = {
                        "flagged_pages": flagged,
                        "pages_evaluated": visual_report.get("pages_evaluated", 0),
                    }
        quality_gate_path = reports_dir / "quality_gate.json"
        log_stage_event(
            "quality_gate",
            "started",
            input_files=[_rel(tex_path)],
        )
        gate_payload = {
            "hallucination": hallucination_report or {},
            "validation": validation_result,
            "visual": visual_report or {},
            "models": {
                "hallucination": HALLUCINATION_MODEL_NAME,
                "validation": VALIDATION_MODEL_NAME,
                "visual": VISUAL_MODEL_NAME,
            },
        }
        quality_gate_path.write_text(json.dumps(gate_payload, indent=2), encoding="utf-8")
        gate_flags = (
            (hallucination_report or {}).get("flagged_count", 0)
            + (hallucination_report or {}).get("claim_flag_count", 0)
            + (visual_report or {}).get("flagged_pages", 0)
        )
        log_stage_event(
            "quality_gate",
            "completed",
            output_files=[_rel(quality_gate_path)],
            notes=f"flags={gate_flags}",
        )
        provenance.record(
            quality_gate_path,
            "quality_gate",
            [HALLUCINATION_MODEL_NAME, VALIDATION_MODEL_NAME, VISUAL_MODEL_NAME],
            "progressive quality validation",
        )
        quality_issues = quality.inspect_tex(tex_path)
        if quality_issues:
            agent_stats["QualityCheck"] = {"issues": quality_issues}
            log_stage_event("quality_check", "failed", notes=_short("; ".join(quality_issues)))
            raise RuntimeError("Quality check failed: " + "; ".join(quality_issues))
        agent_stats["QualityCheck"] = {"issues": 0}
        log_stage_event("quality_check", "completed")
        agent_metrics_path = reports_dir / "agent_metrics.json"
        agent_metrics_path.write_text(json.dumps(agent_stats, indent=2), encoding="utf-8")
        provenance.record(agent_metrics_path, "agent_metrics", [], "stage durations")
        log_stage_event("active_learning", "started")
        try:
            active_summary = active_learning.build_active_learning_queue(
                run_id=run_id,
                chunks_path=chunks_result.chunks_path,
                plan_path=plan_path,
                snippets_path=snippets_path,
                output_dir=reports_dir,
                quality_report=quality_report,
                hallucination_report=hallucination_report,
                gaps_report=gaps_report,
                visual_report=visual_report,
                reward_report=reward_report,
                validation_report=validation_result,
                lint_report=lint_report,
            )
        except Exception as exc:
            log_stage_event("active_learning", "failed", notes=_short(repr(exc)))
        else:
            summary_rel = _rel(active_summary.summary_path)
            queue_rel = _rel(active_summary.queue_path)
            notes = f"candidates={active_summary.summary.get('total_candidates', 0)}"
            log_stage_event(
                "active_learning",
                "completed",
                output_files=[summary_rel, queue_rel],
                notes=notes,
            )
            provenance.record(active_summary.summary_path, "active_learning", [], "active learning summary")
            provenance.record(active_summary.queue_path, "active_learning", [], "active learning queue")
        if missing_chunks:
            raise RuntimeError(
                f"Synthesis missing {missing_count} chunk(s); see {gaps_path} for details."
            )
        clean_release_root_artifacts()
        stage_logger.log("Run", "success", {"output": str(tex_path)})
        total_duration = round(perf_counter() - run_start, 3)
        log_stage_event(
            "run",
            "completed",
            output_files=outputs,
            notes=f"duration={total_duration}s",
        )
        data_logger.set_run_summary(
            {
                "status": "success",
                "input_pdf": _rel(pdf_path),
                "tex_path": _rel(tex_path),
                "pdf_path": outputs[-1] if len(outputs) > 1 else "",
                "artifact_dir": _rel(run_dir),
                "duration_sec": total_duration,
            }
        )
        return tex_path
    except Exception as exc:
        stage_logger.log("Run", "error", {"error": repr(exc)})
        log_stage_event("run", "failed", notes=_short(repr(exc)))
        data_logger.set_run_summary(
            {
                "status": "failed",
                "input_pdf": _rel(pdf_path),
                "artifact_dir": _rel(run_dir),
                "error": _short(repr(exc)),
            }
        )
        raise
    finally:
        provenance_path = reports_dir / "provenance.json"
        try:
            provenance.write(provenance_path, run_id)
        except Exception:
            logging.warning("Failed to write provenance summary to %s", provenance_path, exc_info=True)
        data_logger.write_human_markdown()


def main() -> None:
    args = parse_args()
    if args.benchmark_dir:
        run_benchmark(args)
        return
    tex_path = run_pipeline(args)
    print(f"LaTeX available at {tex_path}")


def run_benchmark(args: argparse.Namespace) -> None:
    pdf_dir = Path(args.benchmark_dir)
    if not pdf_dir.exists():
        raise FileNotFoundError(pdf_dir)
    records = []
    for idx, pdf in enumerate(sorted(pdf_dir.glob("*.pdf"))):
        if idx >= args.benchmark_limit:
            break
        bench_args = argparse.Namespace(
            pdf=str(pdf),
            title=pdf.stem,
            author=args.author,
            run_dir=f"benchmark/{pdf.stem}",
            chunk_chars=args.chunk_chars,
            skip_compile=True,
            log_level=args.log_level,
            benchmark_dir=None,
            benchmark_limit=args.benchmark_limit,
            rag_cache=args.rag_cache,
            rag_refresh=args.rag_refresh,
            reward_mode=args.reward_mode,
            llm_mode=args.llm_mode,
            llm_repo=args.llm_repo,
             llm_backend=args.llm_backend,
             llama_cpp_model=args.llama_cpp_model,
             llama_cpp_grammar=args.llama_cpp_grammar,
            style_domain=args.style_domain,
            no_unicode_sanitizer=args.no_unicode_sanitizer,
        )
        tex_path = run_pipeline(bench_args)
        metrics_path = OUTPUT_DIR / bench_args.run_dir / "reports" / "metrics.json"
        if metrics_path.exists():
            records.append(json.loads(metrics_path.read_text()))
        logging.info("[Benchmark] %s -> %s", pdf.name, tex_path)
    summary = {"documents": len(records), "metrics": records}
    summary_path = OUTPUT_DIR / "benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Benchmark summary saved to {summary_path}")


if __name__ == "__main__":
    main()
