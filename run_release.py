from __future__ import annotations

import argparse
import logging
import json
import os
from datetime import datetime, timezone
from time import perf_counter
from pathlib import Path
from typing import Dict, Any

ROOT = Path(__file__).parent
RELEASE_DIR = ROOT / "release"
INPUT_DIR = RELEASE_DIR / "inputs"
OUTPUT_DIR = RELEASE_DIR / "outputs"
HF_CACHE_DIR = RELEASE_DIR / "models" / "hf_cache"
MODELS_DIR = ROOT / "models"
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


try:  # Optional dependency for memory checkpoints
    import psutil  # type: ignore
except Exception:  # pragma: no cover - psutil always available in release env
    psutil = None  # type: ignore


from release.pipeline import (  # noqa: E402  (import after cache bootstrap)
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
)
from release.pipeline.semantic_chunking import SemanticChunker  # noqa: E402
from release.pipeline.quality_assessment import QualityAssessor  # noqa: E402
from release.pipeline.iterative_refiner import IterativeRefiner  # noqa: E402
from release.models import llm_refiner  # noqa: E402
from release.utils import quality  # noqa: E402


class StageLogger:
    """Append-only checkpoint log that survives terminal crashes."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def _mem_snapshot(self) -> Dict[str, float] | None:
        if psutil is None:
            return None
        try:
            proc = psutil.Process(os.getpid())
            rss = proc.memory_info().rss / 1024**3
            vmem = psutil.virtual_memory()
            return {
                "proc_rss_gb": round(rss, 3),
                "system_used_gb": round(vmem.used / 1024**3, 3),
                "system_total_gb": round(vmem.total / 1024**3, 3),
            }
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
        help="Optional Hugging Face repo id for the LLM refiner (defaults to Qwen/Qwen2.5-1.5B-Instruct)",
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
    run_name = args.run_dir or f"{pdf_path.stem}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(run_name)
    if not run_dir.is_absolute():
        if run_dir.parts and run_dir.parts[0] == "release":
            run_dir = ROOT / run_dir
        else:
            run_dir = OUTPUT_DIR / run_dir
    if run_dir.resolve() == RELEASE_DIR.resolve():
        logging.warning(
            "Run directory %s points to release/. Redirecting artifacts into release/outputs instead.",
            run_dir,
        )
        fallback_name = f"{Path(run_name).stem}_rel"
        run_dir = OUTPUT_DIR / fallback_name
    run_dir.mkdir(parents=True, exist_ok=True)
    agent_stats: Dict[str, Dict[str, float]] = {}
    stage_logger = StageLogger(run_dir / "checkpoint.log")

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
        try:
            llm_refiner_instance = llm_refiner.LLMRefiner(llm_refiner_config)
            stage_logger.log("LLMRefiner", "success")
        except llm_refiner.LLMRefinerError as exc:  # pragma: no cover - heavy dependency
            logging.warning("[refiner] Falling back to template specialists: %s", exc)
            llm_refiner_config = None
            stage_logger.log("LLMRefiner", "error", {"error": str(exc)})
        except Exception as exc:  # pragma: no cover - heavy dependency
            logging.exception("[refiner] Unexpected error while loading LLM refiner, falling back")
            llm_refiner_config = None
            stage_logger.log("LLMRefiner", "error", {"error": repr(exc)})

    stage_logger.log("Run", "start", {"pdf": str(pdf_path)})
    chunker = _configure_chunker(args)
    try:
        chunks_result = run_agent(
            "ParsingAgent",
            ingestion.run_ingestion,
            pdf_path=pdf_path,
            workspace=run_dir / "artifacts",
            chunk_chars=args.chunk_chars,
            ocr_mode="auto",
            capture_page_images=True,
            models_dir=MODELS_DIR,
            semantic_chunker=chunker,
        )
        master_plan_path = run_dir / "master_plan.json"
        title = args.title or pdf_path.stem
        run_agent(
            "PlannerAgent",
            planner.run_planner,
            chunks_result.chunks_path,
            master_plan_path,
            title,
        )
        master_plan = planner.load_master_plan(master_plan_path)
        rag_source_dir = ROOT / "reference_tex"
        rag_cache_path = Path(args.rag_cache) if args.rag_cache else ROOT / "cache" / "rag_index.json"
        if rag_cache_path and not rag_cache_path.is_absolute():
            rag_cache_path = ROOT / rag_cache_path
        rag_cache_path.parent.mkdir(parents=True, exist_ok=True)
        if args.rag_refresh:
            rag.build_index(rag_source_dir, rag_cache_path)
        rag_index = rag.load_or_build_index(rag_source_dir, rag_cache_path)
        plan_path = run_dir / "plan.json"
        run_agent("LayoutAgent", layout.run_layout, chunks_result.chunks_path, plan_path, master_plan_path)
        graph_path = run_dir / "graph.json"
        run_agent("StructureGraphAgent", structure_graph.build_graph, plan_path, chunks_result.chunks_path, graph_path)
        retrieval_path = run_dir / "retrieval.json"
        run_agent("RetrievalAgent", retrieval.build_index, chunks_result.chunks_path, plan_path, retrieval_path)
        snippets_path = run_dir / "snippets.json"
        preamble_path = run_dir / "preamble.json"
        ensure_llm_refiner()
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
        )
        use_sanitizer = not getattr(args, "no_unicode_sanitizer", False)

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
        quality_assessor = QualityAssessor()
        quality_report = run_agent(
            "QualityAssessor",
            quality_assessor.evaluate,
            tex_path,
            chunks_result.chunks_path,
            plan_path,
            snippets_path,
        )
        quality_path = run_dir / "quality_report.json"
        quality_path.write_text(json.dumps(quality_report, indent=2), encoding="utf-8")
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
            refinement = run_agent("IterativeRefiner", refiner.refine)
            tex_path = refinement.tex_path
            quality_report = refinement.report
            quality_path.write_text(json.dumps(quality_report, indent=2), encoding="utf-8")
        agent_stats["QualityScore"] = {"aggregate": quality_report.get("aggregate", 0.0)}
        issues = critique.evaluate_output(plan_path, tex_path)
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
                issues = critique.evaluate_output(plan_path, tex_path)
        if issues:
            logging.warning("Remaining issues after fixes: %s", issues)
        else:
            logging.info("Self-critique passed")
        gaps_report = synthesis_coverage.find_gaps(master_plan_path, snippets_path)
        gaps_path = run_dir / "synthesis_gaps.json"
        gaps_path.write_text(json.dumps(gaps_report, indent=2), encoding="utf-8")
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
        validation_result = run_agent("ValidationAgent", validation.run_validation, tex_path)
        if validation_result["errors"]:
            logging.warning("Validation errors detected: %s", validation_result["errors"])
        metrics_path = run_dir / "metrics.json"
        metrics.evaluate(
            plan_path,
            tex_path,
            retrieval_path,
            metrics_path,
            chunks_result.chunks_path,
            tex_path.parent / "validation.json",
        )
        logging.info("Stage metrics written to %s", metrics_path)
        reward_path = run_dir / "rewards.json"
        reward.evaluate_rewards(
            chunks_result.chunks_path,
            tex_path,
            tex_path.parent / "validation.json",
            reward_path,
            mode=args.reward_mode,
            trace_path=run_dir / "reward_trace.jsonl",
        )
        logging.info("Reward metrics written to %s", reward_path)
        quality_issues = quality.inspect_tex(tex_path)
        if quality_issues:
            agent_stats["QualityCheck"] = {"issues": quality_issues}
            raise RuntimeError("Quality check failed: " + "; ".join(quality_issues))
        agent_stats["QualityCheck"] = {"issues": 0}
        (run_dir / "agent_metrics.json").write_text(json.dumps(agent_stats, indent=2), encoding="utf-8")
        if missing_chunks:
            raise RuntimeError(
                f"Synthesis missing {missing_count} chunk(s); see {gaps_path} for details."
            )
        clean_release_root_artifacts()
        stage_logger.log("Run", "success", {"output": str(tex_path)})
        return tex_path
    except Exception as exc:
        stage_logger.log("Run", "error", {"error": repr(exc)})
        raise


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
        metrics_path = OUTPUT_DIR / bench_args.run_dir / "metrics.json"
        if metrics_path.exists():
            records.append(json.loads(metrics_path.read_text()))
        logging.info("[Benchmark] %s -> %s", pdf.name, tex_path)
    summary = {"documents": len(records), "metrics": records}
    summary_path = OUTPUT_DIR / "benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Benchmark summary saved to {summary_path}")


if __name__ == "__main__":
    main()
