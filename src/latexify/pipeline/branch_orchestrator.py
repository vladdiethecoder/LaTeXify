"""Branch orchestration utilities for LaTeXify's latexify.pipeline."""
from __future__ import annotations

import contextlib
import gc
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional

try:  # pragma: no cover - optional dependency
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

BranchRunner = Callable[["BranchRunContext"], "BranchRunResult"]


@dataclass
class BranchResourceProfile:
    """Resource expectations for a pipeline branch."""

    min_system_memory_gb: float | None = None
    min_gpu_memory_gb: float | None = None
    preferred_device: str | None = None
    env_overrides: Dict[str, str] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if self.min_system_memory_gb is not None:
            payload["min_system_memory_gb"] = float(self.min_system_memory_gb)
        if self.min_gpu_memory_gb is not None:
            payload["min_gpu_memory_gb"] = float(self.min_gpu_memory_gb)
        if self.preferred_device:
            payload["preferred_device"] = self.preferred_device
        if self.env_overrides:
            payload["env_overrides"] = dict(self.env_overrides)
        if self.tags:
            payload["tags"] = list(self.tags)
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass
class BranchSpec:
    name: str
    runner: BranchRunner
    resources: BranchResourceProfile = field(default_factory=BranchResourceProfile)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BranchRunResult:
    branch: str
    status: str
    output_files: List[Path] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    notes: str | None = None
    elapsed_sec: float | None = None


@dataclass
class BranchExecutionSummary:
    total: int = 0
    completed: int = 0
    failed: int = 0
    skipped: int = 0


@dataclass
class BranchRunLedger:
    summary: BranchExecutionSummary
    results: List[BranchRunResult]
    progress_path: Path | None = None


@dataclass
class BranchRunContext:
    name: str
    branch_dir: Path
    report_dir: Path
    shared_context: Dict[str, Any]
    metadata: Dict[str, Any]
    resources: BranchResourceProfile
    _log_hook: Callable[[str, Dict[str, Any]], None]
    _progress_hook: Callable[[str, Dict[str, Any]], None]

    def log(self, status: str, **extra: Any) -> None:
        self._log_hook(status, extra)

    def record_progress(self, status: str, **fields: Any) -> None:
        self._progress_hook(status, fields)

    def get(self, key: str, default: Any = None) -> Any:
        return self.shared_context.get(key, default)

    def ensure_directories(self) -> None:
        self.branch_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)


class BranchOrchestrator:
    """Sequential branch scheduler that is aware of memory budgets."""

    def __init__(
        self,
        *,
        run_id: str,
        run_dir: Path,
        artifacts_dir: Path,
        reports_dir: Path,
        log_event: Callable[[str, str, Any], None],
        shared_context: Dict[str, Any] | None = None,
    ) -> None:
        self.run_id = run_id
        self.run_dir = run_dir
        self.artifacts_dir = artifacts_dir
        self.reports_dir = reports_dir
        self._log_event = log_event
        self._shared_context = dict(shared_context or {})
        self.branch_artifacts_root = artifacts_dir / "branches"
        self.branch_reports_root = reports_dir / "branches"
        self.branch_artifacts_root.mkdir(parents=True, exist_ok=True)
        self.branch_reports_root.mkdir(parents=True, exist_ok=True)
        self._progress_path = reports_dir / "branch_progress.json"
        self._progress_state: Dict[str, Any] = {
            "run_id": run_id,
            "branches": {},
            "updated_at": self._now(),
        }
        self._branches: List[BranchSpec] = []
        self._load_progress()

    def register_branch(self, spec: BranchSpec) -> None:
        entry = self._progress_state.setdefault("branches", {}).setdefault(spec.name, {})
        entry.setdefault("status", "pending")
        entry.setdefault("resources", spec.resources.to_payload())
        entry.setdefault("metadata", dict(spec.metadata))
        entry["branch_dir"] = str(self.branch_artifacts_root / spec.name)
        entry["report_dir"] = str(self.branch_reports_root / spec.name)
        entry["updated_at"] = self._now()
        self._write_progress()
        self._branches.append(spec)

    def run_all(self) -> BranchRunLedger:
        summary = BranchExecutionSummary(total=len(self._branches))
        results: List[BranchRunResult] = []
        if not self._branches:
            return BranchRunLedger(summary=summary, results=results, progress_path=self._progress_path)
        for spec in self._branches:
            context = self._build_context(spec)
            context.ensure_directories()
            memory_snapshot = self._memory_snapshot()
            context.record_progress("resource_check", snapshot=memory_snapshot)
            if not self._has_budget(spec.resources, memory_snapshot):
                note = self._format_memory_note(spec.resources, memory_snapshot)
                result = BranchRunResult(
                    branch=spec.name,
                    status="skipped",
                    metadata={"reason": "insufficient_memory", "snapshot": memory_snapshot},
                    notes=note,
                )
                results.append(result)
                summary.skipped += 1
                self._update_progress(
                    spec.name,
                    "skipped",
                    {
                        "memory": memory_snapshot,
                        "reason": "insufficient_memory",
                    },
                )
                self._log_branch_event(
                    spec.name,
                    "skipped",
                    notes=note,
                    metadata={**spec.metadata, "memory": memory_snapshot},
                )
                continue
            result = self._execute_branch(spec, context)
            results.append(result)
            if result.status == "completed":
                summary.completed += 1
            elif result.status == "skipped":
                summary.skipped += 1
            else:
                summary.failed += 1
        self._write_progress()
        return BranchRunLedger(summary=summary, results=results, progress_path=self._progress_path)

    # Helpers -----------------------------------------------------------------

    def _build_context(self, spec: BranchSpec) -> BranchRunContext:
        branch_dir = self.branch_artifacts_root / spec.name
        report_dir = self.branch_reports_root / spec.name
        return BranchRunContext(
            name=spec.name,
            branch_dir=branch_dir,
            report_dir=report_dir,
            shared_context=self._shared_context,
            metadata=dict(spec.metadata),
            resources=spec.resources,
            _log_hook=lambda status, extra, name=spec.name: self._log_branch_event(name, status, **extra),
            _progress_hook=lambda status, extra, name=spec.name: self._update_progress(name, status, extra),
        )

    def _execute_branch(self, spec: BranchSpec, context: BranchRunContext) -> BranchRunResult:
        env_backup: Dict[str, Optional[str]] = {}
        for key, value in spec.resources.env_overrides.items():
            env_backup[key] = os.environ.get(key)
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        prior_active = os.environ.get("LATEXIFY_ACTIVE_BRANCH")
        os.environ["LATEXIFY_ACTIVE_BRANCH"] = spec.name
        start = perf_counter()
        snapshot_before = self._memory_snapshot()
        self._log_branch_event(
            spec.name,
            "started",
            metadata={**spec.metadata, "resources": spec.resources.to_payload()},
        )
        context.record_progress("running")
        try:
            result = spec.runner(context)
        except Exception as exc:  # pragma: no cover - defensive logging
            logging.exception("Branch %s execution failed", spec.name)
            result = BranchRunResult(
                branch=spec.name,
                status="failed",
                metadata={"error": repr(exc)},
                notes="runtime_error",
            )
        finally:
            if prior_active is not None:
                os.environ["LATEXIFY_ACTIVE_BRANCH"] = prior_active
            else:
                os.environ.pop("LATEXIFY_ACTIVE_BRANCH", None)
            for key, value in env_backup.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            self._flush_caches()
        elapsed = round(perf_counter() - start, 3)
        result.elapsed_sec = elapsed if result.elapsed_sec is None else result.elapsed_sec
        metrics = dict(result.metrics)
        metrics.setdefault("duration_sec", elapsed)
        metrics.setdefault("outputs", float(len(result.output_files)))
        metadata = {**spec.metadata, **result.metadata}
        if snapshot_before:
            metadata.setdefault("resource_usage", {})["start"] = snapshot_before
        snapshot_after = self._memory_snapshot()
        if snapshot_after:
            metadata.setdefault("resource_usage", {})["end"] = snapshot_after
        extra = {
            "output_files": [str(path) for path in result.output_files],
            "metrics": metrics,
            "metadata": metadata,
            "notes": result.notes or metadata.get("description"),
        }
        self._log_branch_event(spec.name, result.status, **extra)
        self._update_progress(
            spec.name,
            result.status,
            {
                "metrics": metrics,
                "metadata": metadata,
                "outputs": extra["output_files"],
            },
        )
        return result

    def _log_branch_event(self, name: str, status: str, **extra: Any) -> None:
        payload = dict(extra)
        payload.setdefault("branch", name)
        output_files = payload.get("output_files")
        if output_files:
            payload["output_files"] = [str(path) for path in output_files]
        metadata = payload.get("metadata") or {}
        metadata.setdefault("branch", name)
        payload["metadata"] = metadata
        self._log_event(f"branch/{name}", status, **payload)

    def _update_progress(self, name: str, status: str, extra: Dict[str, Any] | None) -> None:
        branches = self._progress_state.setdefault("branches", {})
        entry = branches.setdefault(name, {})
        entry["status"] = status
        if extra:
            entry.update(extra)
        entry["updated_at"] = self._now()
        self._progress_state["updated_at"] = self._now()
        self._write_progress()

    def _load_progress(self) -> None:
        if not self._progress_path.exists():
            return
        try:
            data = json.loads(self._progress_path.read_text(encoding="utf-8"))
        except Exception:
            return
        self._progress_state.update({key: value for key, value in data.items() if key != "branches"})
        stored = data.get("branches")
        if isinstance(stored, dict):
            self._progress_state["branches"].update(stored)

    def _write_progress(self) -> None:
        self._progress_state.setdefault("run_id", self.run_id)
        self._progress_state["updated_at"] = self._now()
        payload = json.dumps(self._progress_state, indent=2, ensure_ascii=False)
        self._progress_path.parent.mkdir(parents=True, exist_ok=True)
        self._progress_path.write_text(payload, encoding="utf-8")

    def _memory_snapshot(self) -> Dict[str, Any]:
        snapshot: Dict[str, Any] = {"system_available_gb": None, "gpu": []}
        if psutil is not None:
            try:
                snapshot["system_available_gb"] = round(psutil.virtual_memory().available / 1024**3, 3)
            except Exception:
                snapshot["system_available_gb"] = None
        if torch is not None:
            try:
                available_devices = torch.cuda.device_count()
            except Exception:
                available_devices = 0
            gpu_stats: List[Dict[str, float]] = []
            if torch.cuda.is_available():
                for idx in range(available_devices):
                    try:
                        with torch.cuda.device(idx):
                            free_bytes, total_bytes = torch.cuda.mem_get_info()
                    except Exception:
                        continue
                    gpu_stats.append(
                        {
                            "device": f"cuda:{idx}",
                            "free_gb": round(free_bytes / 1024**3, 3),
                            "total_gb": round(total_bytes / 1024**3, 3),
                        }
                    )
            snapshot["gpu"] = gpu_stats
        return snapshot

    @staticmethod
    def _format_memory_note(resources: BranchResourceProfile, snapshot: Dict[str, Any]) -> str:
        sys_avail = snapshot.get("system_available_gb")
        gpu_avail = next(iter(snapshot.get("gpu") or []), {}).get("free_gb")
        return (
            f"needs>= {resources.min_system_memory_gb}GB sys/{resources.min_gpu_memory_gb}GB gpu "
            f"has {sys_avail}GB sys/{gpu_avail}GB gpu"
        )

    def _has_budget(self, resources: BranchResourceProfile, snapshot: Dict[str, Any]) -> bool:
        sys_ok = True
        gpu_ok = True
        if resources.min_system_memory_gb is not None:
            available = snapshot.get("system_available_gb")
            if available is not None and available < resources.min_system_memory_gb:
                sys_ok = False
        if resources.min_gpu_memory_gb is not None:
            gpu_entries = snapshot.get("gpu") or []
            if not gpu_entries:
                gpu_ok = False
            else:
                gpu_ok = any(entry.get("free_gb", 0.0) >= resources.min_gpu_memory_gb for entry in gpu_entries)
        return sys_ok and gpu_ok

    @staticmethod
    def _flush_caches() -> None:
        gc.collect()
        if torch is not None and torch.cuda.is_available():  # pragma: no cover - device specific
            with contextlib.suppress(Exception):
                torch.cuda.empty_cache()
            with contextlib.suppress(Exception):
                torch.cuda.ipc_collect()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


__all__ = [
    "BranchOrchestrator",
    "BranchRunContext",
    "BranchRunLedger",
    "BranchRunResult",
    "BranchExecutionSummary",
    "BranchResourceProfile",
    "BranchSpec",
]
