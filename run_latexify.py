from __future__ import annotations

import argparse
import logging
import json
import os
import re
import shutil
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from time import perf_counter
from pathlib import Path
from typing import Dict, Any, List, Sequence, Tuple

try:  # pragma: no cover - optional dependency
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

ROOT = Path(__file__).parent
BUILD_ROOT = ROOT / "build"
RELEASE_DIR = ROOT / "src" / "latexify"
INPUT_DIR = RELEASE_DIR / "inputs"
HF_CACHE_DIR = RELEASE_DIR / "models" / "hf_cache"
OUTPUT_DIR = BUILD_ROOT / "runs"
from latexify.core.model_paths import resolve_models_root  # noqa: E402
from latexify.tools import AttemptTracker, ensure_release_dependencies  # noqa: E402

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
    """Force Hugging Face caches into latexify.models/hf_cache if current env is unwritable."""

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

DEPENDENCY_CHECKS = ensure_release_dependencies()

HALLUCINATION_MODEL_NAME = os.environ.get("LATEXIFY_HALLUCINATION_MODEL", "deepseek-ai/DeepSeek-V3")
VALIDATION_MODEL_NAME = os.environ.get("LATEXIFY_VALIDATION_MODEL", "Qwen/Qwen2.5-7B-Instruct")
VISUAL_MODEL_NAME = os.environ.get("LATEXIFY_VISUAL_JUDGE_MODEL", "Qwen/Qwen2.5-VL-32B-Instruct")
VISION_AGENT_VLM_NAME = os.environ.get("LATEXIFY_VISION_AGENT_VLM", "internvl")


try:  # Optional dependency for memory checkpoints
    import psutil  # type: ignore
except Exception:  # pragma: no cover - psutil always available in release env
    psutil = None  # type: ignore


from latexify.pipeline import (  # noqa: E402  (import after cache bootstrap)
    adaptive_quality_gate,
    active_learning,
    assembly,
    branch_outputs,
    branch_evaluator,
    consistency_utils,
    critique,
    hallucination,
    ingestion,
    latex_image_generator,
    layout,
    metrics,
    parallel_branches,
    planner,
    rag,
    retrieval,
    reward_suite,
    snippet_fusion,
    structure_graph,
    synthesis,
    synthesis_coverage,
    validation,
    visual_regression,
    symbolic_render,
    constraint_map_builder,
    flux_inpainting,
)
from latexify.pipeline.branch_orchestrator import BranchRunResult  # noqa: E402
from latexify.pipeline.domain_detector import DomainDetector  # noqa: E402
from latexify.pipeline.semantic_chunking import SemanticChunker  # noqa: E402
from latexify.pipeline.quality_assessment import QualityAssessor  # noqa: E402
from latexify.pipeline.semantic_enricher import SemanticEnricher  # noqa: E402
from latexify.pipeline.iterative_refiner import IterativeRefiner  # noqa: E402
from latexify.pipeline.preamble_optimizer import optimize_preamble  # noqa: E402
from latexify.pipeline.bibliography_utils import generate_bibliography  # noqa: E402
from latexify.pipeline.variable_consistency import normalize_variables  # noqa: E402
from latexify.pipeline.comment_generator import add_section_comments  # noqa: E402
from latexify.pipeline.vision import (
    MultiViewRenderer,
    VisionAgentSuite,
    VisionSynthesisConfig,
)  # noqa: E402
from latexify.models import llm_refiner  # noqa: E402
from latexify.utils import quality  # noqa: E402
from latexify.core import common  # noqa: E402
from latexify.core.config import (  # noqa: E402
    BackendToggleConfig,
    VISION_PRESETS,
    VisionRuntimeConfig,
    BranchRuntimeConfig,
    build_compilation_runtime_config,
    build_kimi_runtime_config,
    build_vision_runtime_config,
    build_branch_runtime_config,
    build_backend_toggle_config,
)
from latexify.pipeline import kimi_metrics  # noqa: E402
from latexify.core.data_pathway_logger import init_logger  # noqa: E402


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


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    return normalized not in {"0", "false", "off", "no"}


def parse_args() -> argparse.Namespace:
    fusion_choices = [choice.value for choice in snippet_fusion.FusionStrategy]
    fusion_aliases = sorted(_FUSION_STRATEGY_ALIASES)
    default_fusion_env = os.environ.get(
        "LATEXIFY_SNIPPET_FUSION_STRATEGY",
        snippet_fusion.FusionStrategy.SELECT_BEST.value,
    )
    try:
        default_fusion_strategy = _resolve_fusion_strategy(default_fusion_env)
    except argparse.ArgumentTypeError:
        logging.warning(
            "Unsupported LATEXIFY_SNIPPET_FUSION_STRATEGY=%s; defaulting to %s",
            default_fusion_env,
            snippet_fusion.FusionStrategy.SELECT_BEST.value,
        )
        default_fusion_strategy = snippet_fusion.FusionStrategy.SELECT_BEST
    default_vision_enabled = _env_flag("LATEXIFY_VISION_SYNTHESIS_ENABLED", True)
    default_vision_preset = os.environ.get("LATEXIFY_VISION_SYNTHESIS_PRESET", "balanced").lower()
    if default_vision_preset not in VISION_PRESETS:
        default_vision_preset = "balanced"
    def _env_float(name: str, default: float) -> float:
        value = os.environ.get(name)
        if value is None:
            return default
        try:
            return float(value)
        except ValueError:
            return default

    def _env_int(name: str, default: int) -> int:
        value = os.environ.get(name)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            return default

    default_kimi_temp = _env_float("LATEXIFY_KIMI_K2_TEMPERATURE", 0.05)
    default_kimi_context = _env_int("LATEXIFY_KIMI_K2_CONTEXT", 32768)
    default_layout_threshold = _env_float("LATEXIFY_LAYOUT_CONFIDENCE_THRESHOLD", 0.0)
    default_retry_count = _env_int("LATEXIFY_COMPILATION_RETRY_COUNT", 3)
    default_robust = _env_flag("LATEXIFY_ENABLE_ROBUST_COMPILATION", True)
    default_monkey = _env_flag("LATEXIFY_ENABLE_MONKEY_OCR", True)

    parser = argparse.ArgumentParser(description="Run the simplified LaTeXify latexify.pipeline.")
    parser.add_argument("--pdf", required=True, help="Path to the draft PDF (absolute or relative to src/latexify/inputs)")
    parser.add_argument("--title", default=None, help="Title for the generated LaTeX document")
    parser.add_argument("--author", default="LaTeXify Release", help="Author name for the document")
    parser.add_argument("--run-dir", default=None, help="Optional output directory for artifacts")
    parser.add_argument("--chunk-chars", type=int, default=ingestion.DEFAULT_CHUNK_CHARS, help="Maximum characters per chunk")
    parser.add_argument(
        "--enable-vision-synthesis",
        dest="enable_vision_synthesis",
        action="store_true",
        default=default_vision_enabled,
        help="Force-enable the MultiViewRenderer + vision agents (env LATEXIFY_VISION_SYNTHESIS_ENABLED).",
    )
    parser.add_argument(
        "--disable-vision-synthesis",
        dest="enable_vision_synthesis",
        action="store_false",
        help="Disable the vision synthesis pipeline regardless of environment toggles.",
    )
    parser.add_argument(
        "--enable-multi-branch",
        dest="enable_multi_branch",
        action="store_true",
        default=None,
        help="Enable the multi-branch orchestrator (env LATEXIFY_ENABLE_MULTI_BRANCH).",
    )
    parser.add_argument(
        "--vision-preset",
        choices=sorted(VISION_PRESETS.keys()),
        default=default_vision_preset,
        help="Vision synthesis preset controlling renderer hyperparameters (env LATEXIFY_VISION_SYNTHESIS_PRESET).",
    )
    parser.add_argument(
        "--kimi-temperature",
        type=float,
        default=default_kimi_temp,
        help="Sampling temperature passed to the Kimi-K2 adapters (env LATEXIFY_KIMI_K2_TEMPERATURE).",
    )
    parser.add_argument(
        "--kimi-context-size",
        type=int,
        default=default_kimi_context,
        help="Context window for Kimi-K2 adapters (env LATEXIFY_KIMI_K2_CONTEXT).",
    )
    parser.add_argument(
        "--layout-confidence-threshold",
        type=float,
        default=default_layout_threshold,
        help="Drop layout regions below this confidence before chunking (env LATEXIFY_LAYOUT_CONFIDENCE_THRESHOLD).",
    )
    parser.add_argument(
        "--layout-backend",
        choices=["pymupdf", "surya"],
        default=os.environ.get("LATEXIFY_LAYOUT_BACKEND", "pymupdf"),
        help="Layout backend used for document segmentation (PyMuPDF heuristics or Surya polygons).",
    )
    parser.add_argument(
        "--disable-surya-math-detector",
        action="store_true",
        help="Disable Surya's math detector even when the Surya backend is active.",
    )
    parser.add_argument(
        "--emit-constraint-maps",
        action="store_true",
        help="Render constraint maps + masks from master OCR items for render-aware reconstruction.",
    )
    parser.add_argument(
        "--constraint-pages",
        default=None,
        help="Optional comma-separated list of page numbers to process when --emit-constraint-maps is set.",
    )
    parser.add_argument(
        "--enable-render-aware",
        action="store_true",
        help="Run Flux-based render-aware reconstruction on pages with constraint maps.",
    )
    parser.add_argument(
        "--render-aware-pages",
        default=None,
        help="Optional comma-separated list of pages to hand to the Flux renderer (default: all).",
    )
    parser.add_argument(
        "--flux-model",
        default=os.environ.get("LATEXIFY_FLUX_MODEL", "black-forest-labs/Flux.1-Fill-dev"),
        help="Diffusion checkpoint (HF repo) for render-aware reconstruction.",
    )
    parser.add_argument(
        "--flux-dtype",
        choices=["fp16", "fp32"],
        default=os.environ.get("LATEXIFY_FLUX_DTYPE", "fp16"),
        help="Computation dtype for the Flux pipeline (fp16 recommended on CUDA).",
    )
    parser.add_argument(
        "--flux-device",
        default=os.environ.get("LATEXIFY_FLUX_DEVICE", "auto"),
        help="Device placement for Flux (auto/cuda/cpu).",
    )
    parser.add_argument(
        "--flux-steps",
        type=int,
        default=int(os.environ.get("LATEXIFY_FLUX_STEPS", "20")),
        help="Number of inference steps per rendered page.",
    )
    parser.add_argument(
        "--flux-guidance",
        type=float,
        default=float(os.environ.get("LATEXIFY_FLUX_GUIDANCE", "1.0")),
        help="Guidance scale applied during diffusion (lower stays closer to constraint map).",
    )
    parser.add_argument(
        "--flux-prompt",
        default=os.environ.get(
            "LATEXIFY_FLUX_PROMPT",
            "High-quality academic page, clean serif typography, precise equations, Springer-style aesthetics.",
        ),
        help="Prompt injected into the Flux diffusion stage.",
    )
    parser.add_argument(
        "--compilation-retry-count",
        type=int,
        default=default_retry_count,
        help="Maximum number of robust compilation retries (env LATEXIFY_COMPILATION_RETRY_COUNT).",
    )
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
        "--fusion-strategy",
        default=default_fusion_strategy,
        type=_resolve_fusion_strategy,
        help=(
            "Snippet fusion strategy (env LATEXIFY_SNIPPET_FUSION_STRATEGY). "
            f"Canonical values: {', '.join(fusion_choices)}. Aliases: {', '.join(fusion_aliases)}."
        ),
    )
    parser.add_argument(
        "--branches",
        default=None,
        help="Comma-separated branch identifiers to run (subset of a,b,c).",
    )
    parser.add_argument(
        "--branch-memory-limit",
        type=float,
        default=None,
        help="Maximum GPU memory (GB) allocated per branch run (env LATEXIFY_BRANCH_MEMORY_LIMIT).",
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
    parser.set_defaults(enable_robust_compilation=default_robust, enable_monkey_ocr=default_monkey)
    parser.add_argument(
        "--enable-robust-compilation",
        dest="enable_robust_compilation",
        action="store_true",
        help="Force-enable the incremental compilation flow (env LATEXIFY_ENABLE_ROBUST_COMPILATION).",
    )
    parser.add_argument(
        "--disable-robust-compilation",
        dest="enable_robust_compilation",
        action="store_false",
        help="Disable robust compilation and use direct tectonic/latexmk execution.",
    )
    parser.add_argument(
        "--enable-monkey-ocr",
        dest="enable_monkey_ocr",
        action="store_true",
        help="Force-enable MonkeyOCR layout analysis (env LATEXIFY_ENABLE_MONKEY_OCR).",
    )
    parser.add_argument(
        "--disable-monkey-ocr",
        dest="enable_monkey_ocr",
        action="store_false",
        help="Disable MonkeyOCR even when the binary is available.",
    )
    return parser.parse_args()


_FUSION_STRATEGY_ALIASES: Dict[str, str] = {
    "confidence": snippet_fusion.FusionStrategy.MERGE_HYBRID.value,
    "rules": snippet_fusion.FusionStrategy.SELECT_BEST.value,
    "llm": snippet_fusion.FusionStrategy.ENSEMBLE_AVERAGE.value,
    "fallback": snippet_fusion.FusionStrategy.ADAPTIVE.value,
    "multi_branch": snippet_fusion.FusionStrategy.MULTI_BRANCH.value,
}


def _resolve_fusion_strategy(value: str | snippet_fusion.FusionStrategy) -> snippet_fusion.FusionStrategy:
    if isinstance(value, snippet_fusion.FusionStrategy):
        return value
    normalized = (value or "").strip().lower()
    normalized = _FUSION_STRATEGY_ALIASES.get(normalized, normalized)
    try:
        return snippet_fusion.FusionStrategy(normalized)
    except ValueError as exc:  # pragma: no cover - argument validation
        valid_options = ", ".join(choice.value for choice in snippet_fusion.FusionStrategy)
        alias_hint = ", ".join(sorted(_FUSION_STRATEGY_ALIASES))
        message = (
            f"Unsupported fusion strategy '{value}'. "
            f"Valid values: {valid_options}. Aliases: {alias_hint or 'none'}."
        )
        raise argparse.ArgumentTypeError(message) from exc


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


def _parse_page_selection(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    if not tokens:
        return None
    if len(tokens) == 1 and tokens[0].lower() == "all":
        return None
    pages: list[int] = []
    for token in tokens:
        if token.lower() == "all":
            return None
        try:
            value = int(token)
        except ValueError:
            logging.warning("Skipping invalid page token '%s' in --constraint-pages.", token)
            continue
        if value > 0:
            pages.append(value)
    return pages or None


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
    chunk_objects: List[common.Chunk] | None = None
    vision_result = None
    vision_report: Dict[str, object] = {}
    vision_branch_data: Dict[str, object] = {}
    latex_image_summary: Dict[str, object] = {}
    vision_runtime = build_vision_runtime_config(
        enabled=args.enable_vision_synthesis,
        preset=args.vision_preset,
    )
    kimi_runtime = build_kimi_runtime_config(
        temperature=args.kimi_temperature,
        context_size=args.kimi_context_size,
    )
    compilation_runtime = build_compilation_runtime_config(
        enable_robust_compilation=args.enable_robust_compilation,
        retry_count=args.compilation_retry_count,
        layout_confidence_threshold=args.layout_confidence_threshold,
        monkey_ocr_enabled=args.enable_monkey_ocr,
    )
    branch_runtime = build_branch_runtime_config(
        enabled=args.enable_multi_branch,
        branches=args.branches,
        memory_limit_gb=args.branch_memory_limit,
    )
    branch_eval_report: Dict[str, object] = {}
    branch_run_results: List[BranchRunResult] = []
    branch_artifacts_manifest: Path | None = None
    branch_performance_metrics: Dict[str, float] = {}
    branch_model_utilization: Dict[str, object] = {}
    branch_fallback_rates: Dict[str, float] = {}
    branch_results_summary: Dict[str, object] = {}
    branch_output_summary: Dict[str, object] = (
        {
            "enabled": branch_runtime.enabled,
            "requested": list(branch_runtime.branches),
        }
        if branch_runtime.enabled
        else {}
    )
    os.environ["LATEXIFY_ENABLE_VISION_SYNTHESIS"] = "1" if vision_runtime.enabled else "0"
    fusion_strategy_arg = getattr(args, "fusion_strategy", snippet_fusion.FusionStrategy.SELECT_BEST)
    fusion_strategy = (
        fusion_strategy_arg
        if isinstance(fusion_strategy_arg, snippet_fusion.FusionStrategy)
        else _resolve_fusion_strategy(str(fusion_strategy_arg))
    )
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

    def _select_artifact(paths: Sequence[Path | str] | None, suffix: str) -> Path | None:
        if not paths:
            return None
        target_suffix = suffix.lower()
        for entry in paths:
            candidate = Path(entry)
            if candidate.suffix.lower() == target_suffix and candidate.exists():
                return candidate
        return None

    def _summarize_branch_telemetry(results: Sequence[BranchRunResult]) -> Tuple[Dict[str, object], Dict[str, float]]:
        model_utilization: Dict[str, object] = {}
        fallback_rates: Dict[str, float] = {}
        for result in results:
            metadata = result.metadata or {}
            models = metadata.get("models")
            if models:
                model_utilization[result.branch] = models
            outputs = float(result.metrics.get("outputs", 0.0)) if result.metrics else 0.0
            failures = float(result.metrics.get("failures", 0.0)) if result.metrics else 0.0
            fallback_rates[result.branch] = round(failures / max(1.0, outputs), 3)
        return model_utilization, fallback_rates

    build_run_dir = BUILD_ROOT / f"run-{run_id}"
    build_run_dir.mkdir(parents=True, exist_ok=True)
    attempt_tracker_config = {
        "pdf": _rel(pdf_path),
        "args": {
            "skip_compile": args.skip_compile,
            "ocr_backend": args.ocr_backend,
            "llm_mode": args.llm_mode,
            "fusion_strategy": str(fusion_strategy),
        },
        "vision": vision_runtime.as_dict(),
        "kimi": kimi_runtime.__dict__,
        "branches": branch_runtime.as_dict(),
        "compilation": {
            "robust": compilation_runtime.enable_robust_compilation,
            "retry_count": compilation_runtime.retry_count,
            "layout_threshold": compilation_runtime.layout_confidence_threshold,
        },
    }
    attempt_tracker = AttemptTracker(run_dir=build_run_dir, run_id=run_id, pdf_path=pdf_path, config=attempt_tracker_config)
    if not AttemptTracker.can_run_next():
        raise RuntimeError(
            "Cumulative runtime across attempts is at or above 7200 seconds. "
            "Stop running additional attempts and summarize current findings."
        )
    attempt_tracker.start()
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
                "fusion_strategy": fusion_strategy.value,
            },
            "vision": vision_runtime.as_dict(),
            "kimi": kimi_runtime.__dict__,
            "compilation": {
                "robust": compilation_runtime.enable_robust_compilation,
                "retry_count": compilation_runtime.retry_count,
                "layout_threshold": compilation_runtime.layout_confidence_threshold,
            },
            "branches": branch_runtime.as_dict(),
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
    backend_config = build_backend_toggle_config(
        ocr_backend=args.ocr_backend,
        layout_backend=args.layout_backend,
        surya_math_detector=not args.disable_surya_math_detector,
        math_ocr_backend=args.math_ocr,
        mineru_enabled=args.ocr_backend == "mineru",
        marker_enabled=args.marker_backup,
        mcp_pdf_processor_enabled=args.mcp_pdf_processor,
        env=os.environ,
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
            notes=(
                f"chunk_chars={args.chunk_chars}, backend={backend_config.ocr_backend}/{ingestion_mode}, "
                f"layout={backend_config.layout_backend}, vision={int(vision_runtime.enabled)}"
            ),
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
                vision_branch_enabled=vision_runtime.enabled,
                layout_confidence_threshold=compilation_runtime.layout_confidence_threshold,
                enable_monkey_ocr=compilation_runtime.monkey_ocr_enabled,
            )
        except Exception as exc:
            log_stage_event("ingestion", "failed", notes=_short(repr(exc)))
            raise
        vision_branch_summary = chunks_result.vision_branch_summary or {}
        if chunk_objects is None:
            chunk_objects = common.load_chunks(chunks_result.chunks_path)
        layout_conf_values = [
            float((chunk.metadata or {}).get("layout_confidence", 0.0))
            for chunk in (chunk_objects or [])
            if (chunk.metadata or {}).get("layout_confidence") is not None
        ]
        if layout_conf_values:
            layout_conf_metrics = {
                "avg": round(sum(layout_conf_values) / len(layout_conf_values), 3),
                "min": round(min(layout_conf_values), 3),
                "threshold": args.layout_confidence_threshold,
            }
        else:
            layout_conf_metrics = {"avg": 0.0, "min": 0.0, "threshold": args.layout_confidence_threshold}
        log_stage_event(
            "ingestion",
            "completed",
            output_files=[
                _rel(chunks_result.chunks_path),
                _rel(artifacts_dir),
            ],
            notes="Workspace artifacts ready.",
            metrics={"vision_branch": vision_branch_summary, "layout_confidence": layout_conf_metrics},
        )
        if vision_branch_summary:
            stage_logger.log("VisionBranching", "summary", vision_branch_summary)
        agent_stats["LayoutAnalysis"] = layout_conf_metrics
        provenance.record(chunks_result.chunks_path, "ingestion", ingestion_models, "chunked text")
        provenance.record(chunks_result.tree_path, "ingestion", ingestion_models, "document tree")
        provenance.record(chunks_result.document_path, "ingestion", ingestion_models, "document representation")
        provenance.record(chunks_result.manifest_path, "ingestion", ingestion_models, "ingestion manifest")
        input_quality_profile: Dict[str, object] = chunks_result.quality_profile or {}
        input_quality_path = reports_dir / "input_quality.json"
        input_quality_path.write_text(json.dumps(input_quality_profile, indent=2), encoding="utf-8")
        if vision_branch_summary:
            vision_branch_data = vision_branch_summary
            agent_stats["VisionBranching"] = {
                "enabled": bool(vision_branch_summary.get("enabled", False)),
                "branches": int(vision_branch_summary.get("branches", 0)),
                "chunk_coverage": float(vision_branch_summary.get("chunk_coverage", 0.0)),
            }
        log_stage_event(
            "input_quality",
            "completed",
            output_files=[_rel(input_quality_path)],
            notes=f"tier={input_quality_profile.get('tier', 'unknown')}",
        )
        provenance.record(input_quality_path, "input_quality", [], "ingestion quality profile")
        quality_profile = input_quality_profile
        constraint_dir = artifacts_dir / "rendered_pages"
        constraint_artifacts: list[constraint_map_builder.ConstraintMapArtifact] = []
        constraint_pages = _parse_page_selection(args.constraint_pages)
        render_aware_pages = _parse_page_selection(args.render_aware_pages)
        builder_page_filter = None
       
        if constraint_pages:
            builder_page_filter = sorted(set(constraint_pages))
        if render_aware_pages:
            builder_page_filter = sorted(
                set(builder_page_filter or []) | set(render_aware_pages)
            ) if builder_page_filter is not None else sorted(set(render_aware_pages))

        need_constraint_maps = args.emit_constraint_maps or args.enable_render_aware
        if need_constraint_maps:
            if not chunks_result.master_ocr_items_path:
                logging.warning("Constraint maps requested but master_ocr_items.json was not emitted by ingestion.")
            else:
                try:
                    formula_cache = artifacts_dir / "formula_cache"
                    renderer = symbolic_render.FormulaRenderer(formula_cache)
                    builder = constraint_map_builder.ConstraintMapBuilder(
                        renderer,
                        page_images_dir=chunks_result.page_images_dir,
                    )
                    constraint_artifacts = builder.build_from_master_items(
                        chunks_result.master_ocr_items_path,
                        constraint_dir,
                        allowed_pages=builder_page_filter,
                    )
                except Exception as exc:
                    logging.warning("Constraint map generation failed: %s", exc, exc_info=True)
                else:
                    summary_payload = [
                        {
                            "page": artifact.page_index,
                            "constraint_map": _rel(artifact.constraint_map),
                            "mask": _rel(artifact.mask_path),
                            "rendered_items": artifact.rendered_items,
                        }
                        for artifact in constraint_artifacts
                    ]
                    summary_path = reports_dir / "constraint_maps.json"
                    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
                    if constraint_artifacts:
                        log_stage_event(
                            "constraint_maps",
                            "completed",
                            output_files=[_rel(summary_path)] + [
                                _rel(artifact.constraint_map) for artifact in constraint_artifacts
                            ],
                            metrics={
                                "pages": len(constraint_artifacts),
                                "rendered_items": sum(a.rendered_items for a in constraint_artifacts),
                            },
                        )
                        agent_stats["ConstraintMaps"] = {
                            "pages": len(constraint_artifacts),
                            "items": sum(a.rendered_items for a in constraint_artifacts),
                        }
                        provenance.record(summary_path, "constraint_maps", [], "render-aware constraint summary")
                    else:
                        log_stage_event(
                            "constraint_maps",
                            "skipped",
                            notes="No renderable regions discovered in master_ocr_items.json",
                        )

        render_summary: list[Dict[str, object]] = []
        if args.enable_render_aware:
            if not constraint_artifacts:
                logging.warning(
                    "Render-aware diffusion requested but constraint maps are missing. "
                    "Pass --emit-constraint-maps or rerun with valid Surya output."
                )
            else:
                target_pages = set(render_aware_pages) if render_aware_pages else None
                flux_cfg = flux_inpainting.FluxConfig(
                    model_id=args.flux_model,
                    dtype=args.flux_dtype,
                    device=args.flux_device,
                    steps=args.flux_steps,
                    guidance=args.flux_guidance,
                    prompt=args.flux_prompt,
                )
                try:
                    flux_engine = flux_inpainting.FluxInpaintingEngine(
                        flux_cfg,
                        workdir=constraint_dir / "renders",
                    )
                    for artifact in constraint_artifacts:
                        if target_pages and artifact.page_index not in target_pages:
                            continue
                        render_path = flux_engine.generate_page(
                            artifact.constraint_map,
                            artifact.mask_path,
                            page_index=artifact.page_index,
                            prompt=args.flux_prompt,
                            steps=args.flux_steps,
                            guidance=args.flux_guidance,
                        )
                        render_summary.append(
                            {
                                "page": artifact.page_index,
                                "constraint_map": _rel(artifact.constraint_map),
                                "mask": _rel(artifact.mask_path),
                                "render": _rel(render_path),
                                "items": artifact.rendered_items,
                            }
                        )
                except Exception as exc:
                    logging.warning("Render-aware diffusion failed: %s", exc, exc_info=True)
                else:
                    if render_summary:
                        render_summary_path = reports_dir / "render_aware.json"
                        render_summary_path.write_text(json.dumps(render_summary, indent=2), encoding="utf-8")
                        log_stage_event(
                            "render_aware",
                            "completed",
                            output_files=[_rel(render_summary_path)]
                            + [entry["render"] for entry in render_summary],
                            metrics={
                                "pages": len(render_summary),
                                "renders": len(render_summary),
                            },
                        )
                        agent_stats["RenderAware"] = {
                            "pages": len(render_summary),
                            "renders": len(render_summary),
                        }
                        provenance.record(render_summary_path, "render_aware", [], "flux render summary")
                    else:
                        log_stage_event(
                            "render_aware",
                            "skipped",
                            notes="No pages met the criteria for render-aware diffusion.",
                        )
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
        branch_ledger = None
        if branch_runtime.enabled:
            branch_shared_context = {
                "chunks_path": chunks_result.chunks_path,
                "plan_path": plan_path,
                "graph_path": graph_path,
                "retrieval_path": retrieval_path,
                "models_dir": MODELS_DIR,
                "run_dir": run_dir,
                "artifacts_dir": artifacts_dir,
                "vision_runtime": vision_runtime.as_dict(),
                "vision_branch_summary": vision_branch_data,
                "branch_c_strategy": snippet_fusion.FusionStrategy.SELECT_BEST.value,
                "preferred_gpu": os.environ.get("LATEXIFY_BRANCH_DEVICE")
                or os.environ.get("LATEXIFY_FORCE_GPU_OCR"),
                "branch_runtime": branch_runtime.as_dict(),
            }
            branch_ledger, branch_artifacts_manifest = parallel_branches.run_parallel_branches(
                run_id=run_id,
                run_dir=run_dir,
                artifacts_dir=artifacts_dir,
                reports_dir=reports_dir,
                shared_context=branch_shared_context,
                log_stage_event=log_stage_event,
                branch_config=branch_runtime,
            )
            if branch_ledger:
                branch_run_results = branch_ledger.results
                branch_metrics = {
                    "branches": float(branch_ledger.summary.total),
                    "completed": float(branch_ledger.summary.completed),
                    "failed": float(branch_ledger.summary.failed),
                    "skipped": float(branch_ledger.summary.skipped),
                }
                branch_model_utilization, branch_fallback_rates = _summarize_branch_telemetry(branch_run_results)
                branch_results_summary = {}
                for result in branch_run_results:
                    entry: Dict[str, object] = {
                        "status": result.status,
                    }
                    if result.metrics:
                        entry["metrics"] = dict(result.metrics)
                    if result.metadata:
                        entry["metadata"] = dict(result.metadata)
                    if result.notes:
                        entry["notes"] = result.notes
                    branch_results_summary[result.branch] = entry
                branch_output_summary.update(
                    {
                        "results": branch_results_summary,
                        "model_utilization_per_branch": branch_model_utilization,
                        "branch_fallback_rates": branch_fallback_rates,
                        "orchestrator_metrics": branch_metrics,
                    }
                )
                if branch_run_results:
                    branch_metrics["max_confidence"] = max(
                        (result.metrics.get("avg_confidence", 0.0) for result in branch_run_results),
                        default=0.0,
                    )
                    branch_metrics["outputs"] = float(
                        sum(result.metrics.get("outputs", 0.0) for result in branch_run_results)
                    )
                agent_stats["BranchOrchestrator"] = branch_metrics
                if branch_artifacts_manifest and branch_artifacts_manifest.exists():
                    log_stage_event(
                        "parallel_branches",
                        "completed",
                        output_files=[_rel(branch_artifacts_manifest)],
                        notes=f"branches={int(branch_metrics['branches'])}",
                        metrics={"branch_orchestrator": branch_metrics},
                        metadata={"selected": list(branch_runtime.branches)},
                    )
                    provenance.record(
                        branch_artifacts_manifest,
                        "parallel_branches",
                        [],
                        "branch artifact manifest",
                    )
        latex_images_dir = artifacts_dir / "latex_images"
        log_stage_event(
            "latex_image_generation",
            "started",
            input_files=[_rel(chunks_result.chunks_path)],
        )
        try:
            if chunk_objects is None:
                chunk_objects = common.load_chunks(chunks_result.chunks_path)
            image_generator = latex_image_generator.LaTeXImageGenerator(max_regenerations=3)
            latex_image_summary = image_generator.generate(chunk_objects, latex_images_dir)
        except Exception as exc:
            log_stage_event("latex_image_generation", "failed", notes=_short(repr(exc)))
        else:
            common.save_chunks(chunk_objects, chunks_result.chunks_path)
            log_stage_event(
                "latex_image_generation",
                "completed",
                output_files=[_rel(latex_images_dir)],
                metrics=latex_image_summary.get("metrics", {}),
            )
            provenance.record(
                latex_images_dir,
                "latex_image_generation",
                [],
                "latex image previews",
            )
        vision_views_dir = artifacts_dir / "vision_views"
        if vision_runtime.enabled:
            log_stage_event(
                "vision_synthesis",
                "started",
                input_files=[_rel(chunks_result.chunks_path)],
                notes=f"preset={vision_runtime.preset}",
            )
            try:
                chunk_objects = common.load_chunks(chunks_result.chunks_path)
                renderer_defaults = asdict(VisionSynthesisConfig())
                renderer_defaults.update(vision_runtime.resolved_overrides())
                renderer_config = VisionSynthesisConfig(**renderer_defaults)
                renderer = MultiViewRenderer(renderer_config, output_dir=vision_views_dir)
                vision_result = run_agent(
                    "MultiViewRenderer",
                    renderer.render,
                    chunk_objects,
                    output_dir=vision_views_dir,
                )
            except Exception as exc:
                log_stage_event("vision_synthesis", "failed", notes=_short(repr(exc)))
                raise
            chunk_objects = vision_result.attach_metadata(chunk_objects)
            common.save_chunks(chunk_objects, chunks_result.chunks_path)
            vision_summary = vision_result.summary()
            log_stage_event(
                "vision_synthesis",
                "completed",
                output_files=[_rel(chunks_result.chunks_path), _rel(vision_views_dir)],
                notes=f"chunks={vision_summary['chunks']},views={vision_summary['views']}",
                metrics={"vision_synthesis": vision_summary | {"preset": vision_runtime.preset}},
            )
            stage_logger.log(
                "VisionSynthesis",
                "summary",
                {"preset": vision_runtime.preset, **vision_summary},
            )
            provenance.record(vision_views_dir, "vision_synthesis", [], "vision crops")
            if "MultiViewRenderer" in agent_stats:
                agent_stats["MultiViewRenderer"].update(
                    {
                        "chunks": float(vision_summary.get("chunks", 0)),
                        "views": float(vision_summary.get("views", 0)),
                        "avg_views_per_chunk": float(vision_summary.get("avg_views_per_chunk", 0.0)),
                        "preset": vision_runtime.preset,
                    }
                )
            else:
                agent_stats["MultiViewRenderer"] = {
                    "chunks": float(vision_summary.get("chunks", 0)),
                    "views": float(vision_summary.get("views", 0)),
                    "avg_views_per_chunk": float(vision_summary.get("avg_views_per_chunk", 0.0)),
                    "preset": vision_runtime.preset,
                }
            vision_report_path = reports_dir / "vision_diagnostics.json"
            log_stage_event(
                "vision_agents",
                "started",
                input_files=[_rel(chunks_result.chunks_path)],
                notes="VisionAgentSuite early pass",
            )
            try:
                def _run_vision_agents() -> Dict[str, object]:
                    if chunk_objects is None or vision_result is None:
                        return {
                            "summary": {"chunks": 0, "agents": [], "avg_confidence": 0.0},
                            "chunks": {},
                        }
                    suite = VisionAgentSuite()
                    chunk_payload: Dict[str, Dict[str, Dict[str, object]]] = {}
                    total_conf = 0.0
                    total_results = 0
                    for chunk in chunk_objects:
                        views = vision_result.views_by_chunk.get(chunk.chunk_id, [])
                        if not views:
                            continue
                        results = suite.evaluate(chunk, views)
                        if not results:
                            continue
                        total_results += len(results)
                        total_conf += sum(entry.confidence for entry in results)
                        metadata = dict(chunk.metadata or {})
                        scores = dict(metadata.get("vision_scores") or {})
                        notes_map = dict(metadata.get("vision_notes") or {})
                        chunk_entry: Dict[str, Dict[str, object]] = {}
                        for result in results:
                            chunk_entry[result.agent] = {
                                "confidence": result.confidence,
                                "summary": result.summary,
                                "metadata": result.metadata,
                            }
                            scores[result.agent] = result.confidence
                            notes_map[result.agent] = result.summary
                        metadata["vision_scores"] = scores
                        metadata["vision_notes"] = notes_map
                        chunk.metadata = metadata
                        chunk_payload[chunk.chunk_id] = chunk_entry
                    if chunk_payload:
                        common.save_chunks(chunk_objects, chunks_result.chunks_path)
                    avg_conf = total_conf / total_results if total_results else 0.0
                    summary = {
                        "chunks": len(chunk_payload),
                        "agents": [agent.name for agent in suite.agents],
                        "avg_confidence": round(avg_conf, 3),
                    }
                    return {"summary": summary, "chunks": chunk_payload}

                vision_report = run_agent("VisionAgentSuite", _run_vision_agents)
            except Exception as exc:
                log_stage_event("vision_agents", "failed", notes=_short(repr(exc)))
                raise
            vision_report_path.write_text(json.dumps(vision_report, indent=2), encoding="utf-8")
            vision_summary_payload = vision_report.get("summary", {})
            chunk_count = int(vision_summary_payload.get("chunks", 0))
            avg_conf = float(vision_summary_payload.get("avg_confidence", 0.0))
            log_stage_event(
                "vision_agents",
                "completed",
                output_files=[_rel(vision_report_path)],
                notes=f"chunks={chunk_count},avg={avg_conf:.2f}",
                metrics={
                    "vision_agents": {
                        "chunks": chunk_count,
                        "avg_confidence": avg_conf,
                    }
                },
            )
            provenance.record(vision_report_path, "vision_agents", [], "vision diagnostics")
            if "VisionAgentSuite" in agent_stats:
                agent_stats["VisionAgentSuite"].update(
                    {
                        "chunks": float(chunk_count),
                        "avg_confidence": avg_conf,
                    }
                )
            else:
                agent_stats["VisionAgentSuite"] = {"chunks": float(chunk_count), "avg_confidence": avg_conf}
        else:
            log_stage_event(
                "vision_synthesis",
                "skipped",
                notes="vision synthesis disabled",
            )
            stage_logger.log("VisionSynthesis", "skipped", {"enabled": False})
            log_stage_event(
                "vision_agents",
                "skipped",
                notes="vision synthesis disabled",
            )
        vision_result = None
        chunk_objects = None
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
                refinement_passes=args.compilation_retry_count,
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
        compilation_metrics = assembly.consume_compilation_metrics()
        if compilation_metrics:
            agent_stats["RobustCompilation"] = {
                "attempts": float(compilation_metrics.get("compilation_attempts", 0) or 0),
                "recovery_success": 1.0 if compilation_metrics.get("recovery_success") else 0.0,
            }
            log_stage_event(
                "robust_compilation",
                "completed" if compilation_metrics.get("compilation_attempts") else "skipped",
                metrics=compilation_metrics,
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
                max_iterations=args.compilation_retry_count,
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
        fusion_report: Dict[str, object] = {}
        snippet_fusion_path = reports_dir / "snippet_fusion.json"
        log_stage_event(
            "snippet_fusion",
            "started",
            input_files=[_rel(snippets_path)],
            notes=f"strategy={fusion_strategy.value}",
        )
        try:
            fusion_report = snippet_fusion.run_snippet_fusion(
                chunks_result.chunks_path,
                snippets_path,
                metrics_path,
                validation_path,
                snippet_fusion_path,
                strategy=fusion_strategy,
                branch_manifest_path=branch_artifacts_manifest,
            )
        except Exception as exc:
            log_stage_event("snippet_fusion", "failed", notes=_short(repr(exc)))
            fusion_report = {}
        else:
            log_stage_event(
                "snippet_fusion",
                "completed",
                output_files=[_rel(snippet_fusion_path)],
                notes=f"avg={fusion_report.get('aggregate_confidence', 0.0):.2f}",
                metrics={
                    "snippet_fusion": {
                        "strategy": fusion_strategy.value,
                        "avg_confidence": fusion_report.get("aggregate_confidence", 0.0),
                        "flagged": len(fusion_report.get("flagged_chunks", [])),
                    },
                    "fusion_effectiveness_scores": {
                        "avg_confidence": fusion_report.get("aggregate_confidence", 0.0),
                        "flagged": len(fusion_report.get("flagged_chunks", [])),
                    },
                },
            )
            provenance.record(snippet_fusion_path, "snippet_fusion", [], "snippet scoring")
        if branch_runtime.enabled and branch_run_results:
            branch_eval_path = reports_dir / "branch_evaluation.json"
            log_stage_event("branch_evaluation", "started")
            try:
                branch_eval_report = branch_evaluator.evaluate_branches(
                    chunks_result.chunks_path,
                    snippets_path,
                    branch_artifacts_manifest,
                    latex_image_summary,
                    vision_branch_data,
                    branch_eval_path,
                )
            except Exception as exc:
                branch_eval_report = {}
                log_stage_event("branch_evaluation", "failed", notes=_short(repr(exc)))
            else:
                if not isinstance(branch_eval_report, dict):
                    logging.warning(
                        "Branch evaluator returned %s; wrapping for telemetry.",
                        type(branch_eval_report).__name__,
                    )
                    branch_eval_report = {"raw": branch_eval_report}
                log_stage_event(
                    "branch_evaluation",
                    "completed",
                    output_files=[_rel(branch_eval_path)],
                    metrics=branch_eval_report.get("metrics", {}),
                )
                provenance.record(branch_eval_path, "branch_evaluation", [], "branch comparison report")
                agent_stats["BranchEvaluation"] = branch_eval_report.get("metrics", {})
                if isinstance(quality_profile, dict):
                    quality_profile.setdefault("branch_metrics", branch_eval_report.get("metrics", {}))
                branch_performance_metrics = branch_eval_report.get("metrics", {})
                if branch_output_summary:
                    branch_output_summary.setdefault(
                        "branch_performance_metrics",
                        branch_performance_metrics,
                    )
        fusion_effectiveness = {
            "avg_confidence": fusion_report.get("aggregate_confidence", 0.0),
            "flagged_chunks": len(fusion_report.get("flagged_chunks", [])),
        }
        agent_stats["SnippetFusion"] = {
            "strategy": fusion_strategy.value,
            **fusion_effectiveness,
        }
        agent_stats["FusionEffectiveness"] = fusion_effectiveness
        consistency_report_path = reports_dir / "consistency.json"
        try:
            consistency_payload = consistency_utils.visual_textual_consistency(
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
            reward_suite.evaluate_rewards(
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
        cross_validation_report = reward_report.get("cross_validation") or {}
        if cross_validation_report:
            cv_metrics = {
                "overall_score": cross_validation_report.get("overall_score", 0.0),
                "confidence": cross_validation_report.get("confidence", 0.0),
                "semantic": (cross_validation_report.get("semantic") or {}).get("composite", 0.0),
                "content": (cross_validation_report.get("content") or {}).get("composite", 0.0),
                "structural": (cross_validation_report.get("structural") or {}).get("composite", 0.0),
                "visual": (cross_validation_report.get("visual") or {}).get("composite", 0.0),
            }
            agent_stats["CrossValidation"] = {
                "overall": cv_metrics["overall_score"],
                "confidence": cv_metrics["confidence"],
            }
            log_stage_event(
                "cross_validation",
                "completed",
                metrics=cv_metrics,
            )
        visual_report = None
        quality_gate_report = None
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
        doc_profile = adaptive_quality_gate.infer_document_profile(chunks_result.chunks_path, plan_path)
        gate_evaluator = adaptive_quality_gate.AdaptiveQualityGate(doc_profile)
        gate_metrics = {
            "hallucination": hallucination_report,
            "validation": validation_result,
            "visual": visual_report,
            "reward": reward_report,
            "cross_validation": reward_report.get("cross_validation"),
            "input_profile": input_quality_profile,
        }
        gate_result = gate_evaluator.evaluate(gate_metrics)
        branch_summary_payload = {
            "enabled": vision_runtime.enabled,
            "preset": vision_runtime.preset,
            "strategy": input_quality_profile.get("branch_strategy"),
            "signal": input_quality_profile.get("vision_signal"),
            "consistency": input_quality_profile.get("branch_consistency"),
            "coverage": vision_branch_data.get("chunk_coverage"),
            "branches": vision_branch_data.get("branches"),
            "fusion_strategy": fusion_strategy.value,
        }
        gate_payload = gate_result.payload
        gate_payload["vision_branch"] = branch_summary_payload
        if isinstance(branch_eval_report, dict) and branch_eval_report:
            gate_payload["branch_evaluation"] = branch_eval_report
        if branch_output_summary:
            gate_payload["branch_outputs"] = branch_output_summary
        gate_payload["models"] = {
            "hallucination": HALLUCINATION_MODEL_NAME,
            "validation": VALIDATION_MODEL_NAME,
            "visual": VISUAL_MODEL_NAME,
            "vision": VISION_AGENT_VLM_NAME,
        }
        snippet_entry = fusion_report or {}
        snippet_entry.setdefault("strategy", fusion_strategy.value)
        gate_payload["snippet_fusion"] = snippet_entry
        adaptive_quality_gate.save_gate_payload(quality_gate_path, gate_payload)
        status = "completed"
        notes = "passed"
        if gate_result.failed_dimensions and not gate_result.overrides_applied:
            status = "failed"
            notes = f"failed={','.join(gate_result.failed_dimensions)}"
        elif gate_result.failed_dimensions and gate_result.overrides_applied:
            status = "override"
            notes = f"override={','.join(gate_result.failed_dimensions)}"
        gate_stage_metrics: Dict[str, object] = {}
        if branch_performance_metrics:
            gate_stage_metrics["branch_performance_metrics"] = branch_performance_metrics
        if branch_model_utilization:
            gate_stage_metrics["model_utilization_per_branch"] = branch_model_utilization
        if branch_fallback_rates:
            gate_stage_metrics["branch_fallback_rates"] = branch_fallback_rates
        if fusion_effectiveness:
            gate_stage_metrics["fusion_effectiveness_scores"] = fusion_effectiveness
        log_kwargs: Dict[str, object] = {
            "output_files": [_rel(quality_gate_path)],
            "notes": _short(notes),
        }
        if gate_stage_metrics:
            log_kwargs["metrics"] = gate_stage_metrics
        log_stage_event("quality_gate", status, **log_kwargs)
        provenance.record(
            quality_gate_path,
            "quality_gate",
            [HALLUCINATION_MODEL_NAME, VALIDATION_MODEL_NAME, VISUAL_MODEL_NAME],
            "progressive quality validation",
        )
        quality_gate_report = gate_payload
        agent_stats["QualityGate"] = {
            "passed": gate_result.passed,
            "failed_dimensions": gate_result.failed_dimensions,
            "overrides": gate_result.overrides_applied,
        }
        if not gate_result.passed and not gate_result.overrides_applied:
            raise RuntimeError(
                "Adaptive quality gate failed: " + ", ".join(gate_result.failed_dimensions or ["unknown"])
            )
        quality_issues = quality.inspect_tex(tex_path)
        if quality_issues:
            agent_stats["QualityCheck"] = {"issues": quality_issues}
            log_stage_event("quality_check", "failed", notes=_short("; ".join(quality_issues)))
            raise RuntimeError("Quality check failed: " + "; ".join(quality_issues))
        agent_stats["QualityCheck"] = {"issues": 0}
        log_stage_event("quality_check", "completed")
        branch_output_manifest_path = None
        branch_metrics_summary: Dict[str, Dict[str, object]] = {}
        if branch_run_results:
            output_manager = branch_outputs.BranchOutputManager(
                run_dir=run_dir,
                reports_dir=reports_dir,
                rel_path_fn=_rel,
            )
            pdf_source = pdf_candidate if (not args.skip_compile and pdf_candidate.exists()) else None
            for result in branch_run_results:
                entry_metadata = dict(result.metadata or {})
                if result.notes:
                    entry_metadata.setdefault("notes", result.notes)
                tex_candidate = _select_artifact(result.output_files, suffix=".tex") or tex_path
                pdf_candidate_path = pdf_source or _select_artifact(result.output_files, suffix=".pdf")
                output_manager.register_branch_output(
                    name=result.branch,
                    tex_source=tex_candidate,
                    pdf_source=pdf_candidate_path,
                    status=result.status,
                    metadata=entry_metadata,
                )
            best_branch = branch_outputs.select_best_branch(branch_run_results)
            legacy_pdf_target = pdf_candidate if (pdf_candidate.exists() and not args.skip_compile) else None
            branch_output_manifest_path = output_manager.finalize(
                best_branch=best_branch,
                legacy_tex=tex_path,
                legacy_pdf=legacy_pdf_target,
            )
            branch_metrics_summary = output_manager.metrics_summary()
            if branch_output_summary:
                branch_output_summary.setdefault("results", branch_metrics_summary)
                branch_output_summary["artifacts"] = branch_metrics_summary
                branch_output_summary["manifest"] = _rel(branch_output_manifest_path)
                branch_output_summary["best_branch"] = best_branch
            log_stage_event(
                "branch_outputs",
                "completed",
                output_files=[_rel(branch_output_manifest_path)],
                notes=f"best={best_branch}",
                metrics={
                    "branch_outputs": branch_metrics_summary,
                    "branch_performance_metrics": branch_performance_metrics,
                    "model_utilization_per_branch": branch_model_utilization,
                    "branch_fallback_rates": branch_fallback_rates,
                    "fusion_effectiveness_scores": fusion_effectiveness,
                },
                metadata={"branches": list(branch_runtime.branches)},
            )
            provenance.record(
                branch_output_manifest_path,
                "branch_outputs",
                [],
                "branch output manifest",
            )
            agent_stats["BranchPerformance"] = branch_performance_metrics
            agent_stats["FusionEffectiveness"] = fusion_effectiveness
        kimi_snapshot = kimi_metrics.snapshot()
        if kimi_snapshot.get("calls", 0):
            calls = kimi_snapshot.get("calls", 0.0)
            total_time = kimi_snapshot.get("total_time", 0.0)
            average_ms = round((total_time / calls) * 1000.0, 2) if calls else 0.0
            repair_attempts = kimi_snapshot.get("repair_attempts", 0.0) or 0.0
            repair_rate = (
                kimi_snapshot.get("repair_success", 0.0) / repair_attempts if repair_attempts else 0.0
            )
            kimi_payload = {
                "calls": calls,
                "avg_time_ms": average_ms,
                "success_rate": round((kimi_snapshot.get("success", 0.0) / calls) if calls else 0.0, 3),
                "repair_success_rate": round(repair_rate, 3),
            }
            agent_stats["KimiK2"] = kimi_payload
            log_stage_event("kimi_k2", "completed", metrics=kimi_payload)
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
                quality_gate_report=quality_gate_report,
                branch_report=branch_eval_report,
                branch_outputs=branch_output_summary,
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
        attempt_tracker.finish(
            status="success",
            classification="technical_success",
            duration_sec=total_duration,
            outputs=outputs,
            logs=[_rel(logs_dir / "checkpoint.log")],
            notes="pipeline completed",
        )
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
        elapsed = round(perf_counter() - run_start, 3)
        attempt_tracker.finish(
            status="failed",
            classification="technical_failure",
            duration_sec=elapsed,
            outputs=outputs if "outputs" in locals() else None,
            logs=[_rel(logs_dir / "checkpoint.log")],
            notes=_short(repr(exc)),
        )
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
        bench_kwargs = vars(args).copy()
        bench_kwargs.update(
            {
                "pdf": str(pdf),
                "title": pdf.stem,
                "author": args.author,
                "run_dir": f"benchmark/{pdf.stem}",
                "skip_compile": True,
                "benchmark_dir": None,
                "benchmark_limit": args.benchmark_limit,
            }
        )
        bench_args = argparse.Namespace(**bench_kwargs)
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
