"""Sequential branch execution utilities for the release pipeline."""
from __future__ import annotations

import contextlib
import gc
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

try:  # pragma: no cover - optional dependency
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

from .branch_orchestrator import (
    BranchOrchestrator,
    BranchResourceProfile,
    BranchRunContext,
    BranchRunLedger,
    BranchRunResult,
    BranchSpec,
)
from . import branch_b_vision
from ..core import common
from ..core.config import BranchRuntimeConfig
from ..models.model_adapters import release_shared_adapter

LOGGER = logging.getLogger(__name__)


@dataclass
class BranchProfile:
    """Static configuration for a logical branch."""

    name: str
    description: str
    resources: BranchResourceProfile
    metadata: Dict[str, Any]


def run_parallel_branches(
    *,
    run_id: str,
    run_dir: Path,
    artifacts_dir: Path,
    reports_dir: Path,
    shared_context: Dict[str, Any],
    log_stage_event,
    branch_config: BranchRuntimeConfig | None = None,
) -> Tuple[BranchRunLedger | None, Path | None]:
    """Execute branches sequentially and return the ledger + manifest path."""

    if branch_config is None or not branch_config.enabled:
        return None, None
    if not shared_context.get("chunks_path"):
        LOGGER.debug("parallel_branches: chunks_path missing; skipping branch execution.")
        return None, None
    orchestrator = BranchOrchestrator(
        run_id=run_id,
        run_dir=run_dir,
        artifacts_dir=artifacts_dir,
        reports_dir=reports_dir,
        log_event=log_stage_event,
        shared_context=shared_context,
    )
    for spec in _build_branch_specs(shared_context, branch_config):
        orchestrator.register_branch(spec)
    ledger = orchestrator.run_all()
    manifest_path = _write_branch_manifest(ledger, reports_dir)
    return ledger, manifest_path


def _build_branch_specs(shared_context: Dict[str, Any], branch_config: BranchRuntimeConfig) -> Iterable[BranchSpec]:
    selected = set(branch_config.branches)
    memory_limit = branch_config.memory_limit_gb
    profiles: List[Tuple[BranchProfile, Any]] = [
        (
            BranchProfile(
                name="branch_a",
                description="Primary OCR ensemble snapshot",
                resources=BranchResourceProfile(
                    min_system_memory_gb=4.0,
                    min_gpu_memory_gb=0.0,
                    metadata={"sequence": 1},
                ),
                metadata={"profile": "ocr_ensemble", "models": ["pypdf"], "branch": "branch_a"},
            ),
            _wrap_runner(_run_branch_a),
        ),
        (
            BranchProfile(
                name="branch_b",
                description="Nougat + vision synthesis",
                resources=BranchResourceProfile(
                    min_system_memory_gb=8.0,
                    min_gpu_memory_gb=3.5,
                    preferred_device=shared_context.get("preferred_gpu"),
                    env_overrides={"LATEXIFY_BRANCH_MODE": "vision"},
                    metadata={"sequence": 2},
                ),
                metadata={"profile": "vision_synthesis", "models": ["nougat", "pix2tex", "internvl"], "branch": "branch_b"},
            ),
            _wrap_runner(branch_b_vision.run_branch),
        ),
        (
            BranchProfile(
                name="branch_c",
                description="Lightweight fusion planning",
                resources=BranchResourceProfile(
                    min_system_memory_gb=2.0,
                    min_gpu_memory_gb=0.0,
                    metadata={"sequence": 3},
                ),
                metadata={"profile": "fusion_light", "models": ["fusion_engine"], "branch": "branch_c"},
            ),
            _wrap_runner(_run_branch_c),
        ),
    ]
    for profile, runner in profiles:
        if profile.name not in selected:
            continue
        if memory_limit is not None and profile.resources.min_gpu_memory_gb:
            profile.resources.min_gpu_memory_gb = min(profile.resources.min_gpu_memory_gb, memory_limit)
            profile.metadata.setdefault("resource_limit_gb", memory_limit)
        yield BranchSpec(
            name=profile.name,
            runner=runner,
            resources=profile.resources,
            metadata={"description": profile.description, **profile.metadata},
        )


def _run_branch_a(context: BranchRunContext) -> BranchRunResult:
    chunks_path_value = context.get("chunks_path")
    if not chunks_path_value:
        return BranchRunResult(branch=context.name, status="skipped", notes="chunks missing")
    chunks_path = Path(chunks_path_value)
    if not chunks_path.exists():
        return BranchRunResult(branch=context.name, status="skipped", notes="chunks missing")
    artifact_dir = context.branch_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    target_path = artifact_dir / "chunks.json"
    shutil.copy2(chunks_path, target_path)
    chunks = common.load_chunks(chunks_path)
    chunk_count = len(chunks)
    avg_chars = 0.0
    if chunk_count:
        avg_chars = round(sum(len(chunk.text or "") for chunk in chunks) / chunk_count, 3)
    metadata = {
        "chunks": chunk_count,
        "avg_chunk_chars": avg_chars,
        "source": str(chunks_path),
        "models": ["pypdf"],
    }
    metrics = {"chunks": float(chunk_count), "avg_chars": avg_chars}
    context.log(
        "persisted",
        notes=f"snapshot={chunk_count} chunks",
        output_files=[str(target_path)],
        metrics=metrics,
    )
    return BranchRunResult(
        branch=context.name,
        status="completed",
        output_files=[target_path],
        metrics=metrics,
        metadata=metadata,
        notes="ocr_snapshot",
    )


def _run_branch_c(context: BranchRunContext) -> BranchRunResult:
    strategy = context.get("branch_c_strategy", "select_best")
    payload = {
        "strategy": strategy,
        "notes": "Lightweight fusion branch prefers minimal blending.",
    }
    target_path = context.branch_dir / "fusion_profile.json"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    metrics = {"lightweight": 1.0 if strategy == "select_best" else 0.5}
    context.log(
        "configured",
        notes=f"fusion_strategy={strategy}",
        output_files=[str(target_path)],
        metrics=metrics,
    )
    return BranchRunResult(
        branch=context.name,
        status="completed",
        output_files=[target_path],
        metrics=metrics,
        metadata={"fusion_strategy": strategy, "models": ["fusion_engine"]},
        notes="fusion_profile",
    )


def _write_branch_manifest(ledger: BranchRunLedger | None, reports_dir: Path) -> Path | None:
    if ledger is None:
        return None
    manifest_path = reports_dir / "branch_artifacts.json"
    summary = {
        "summary": {
            "total": ledger.summary.total,
            "completed": ledger.summary.completed,
            "failed": ledger.summary.failed,
            "skipped": ledger.summary.skipped,
        },
        "results": [
            {
                "branch": result.branch,
                "status": result.status,
                "output_files": [str(path) for path in result.output_files],
                "metrics": result.metrics,
                "metadata": result.metadata,
                "notes": result.notes,
            }
            for result in ledger.results
        ],
    }
    manifest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return manifest_path


def _wrap_runner(fn):
    def _runner(context: BranchRunContext) -> BranchRunResult:
        try:
            return fn(context)
        finally:
            _force_model_cleanup()
    return _runner


def _force_model_cleanup() -> None:
    for adapter in ("internvl", "florence2", "nougat"):
        with contextlib.suppress(Exception):
            release_shared_adapter(adapter)
    gc.collect()
    if torch is not None and torch.cuda.is_available():  # pragma: no cover - hardware dependent
        with contextlib.suppress(Exception):
            torch.cuda.empty_cache()
        with contextlib.suppress(Exception):
            torch.cuda.ipc_collect()


__all__ = ["run_parallel_branches"]
