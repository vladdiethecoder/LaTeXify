"""
Attempt tracking utilities for release runs.

Each pipeline invocation logs a structured record in both the per-run directory
and a global ledger (build/attempt_ledger.json) to help correlate failures,
quality issues, and cumulative runtime.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_ROOT = REPO_ROOT / "build"
LEDGER_PATH = BUILD_ROOT / "attempt_ledger.json"
RUNTIME_CAP_SECONDS = 7200


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class AttemptRecord:
    attempt_id: int
    pdf: str
    config: Dict[str, Any]
    started_at: str
    run_id: str
    run_dir: str
    status: str | None = None
    classification: str | None = None  # technical_failure, technical_success, poor_quality
    duration_sec: float | None = None
    cumulative_runtime_sec: float | None = None
    pdf_runtime_sec: float | None = None
    notes: str | None = None
    outputs: List[str] | None = None
    logs: List[str] | None = None


class AttemptTracker:
    """
    Maintains attempt history for release runs.

    Global ledger is keyed by attempt_id; per-PDF counters are tracked separately
    so diagnostics can see how many tries each document required.
    """

    def __init__(self, run_dir: Path, run_id: str, pdf_path: Path, config: Dict[str, Any]) -> None:
        self.run_dir = run_dir
        self.run_id = run_id
        self.pdf_path = pdf_path
        self.config = config
        self._ledger = self._load_ledger()
        self._attempt_id = len(self._ledger.get("attempts", [])) + 1
        self._pdf_attempt_id = self._ledger.get("pdf_attempts", {}).get(str(pdf_path), 0) + 1
        self._record = AttemptRecord(
            attempt_id=self._attempt_id,
            pdf=str(pdf_path),
            config=config,
            started_at=_timestamp(),
            run_id=run_id,
            run_dir=str(run_dir),
        )

    @property
    def attempt_id(self) -> int:
        return self._attempt_id

    @property
    def pdf_attempt_id(self) -> int:
        return self._pdf_attempt_id

    def start(self) -> None:
        self._append_record({"status": "running"})

    def finish(
        self,
        *,
        status: str,
        classification: str,
        duration_sec: float,
        outputs: Optional[List[str]] = None,
        logs: Optional[List[str]] = None,
        notes: str | None = None,
    ) -> None:
        cumulative_runtime = self._ledger.get("cumulative_runtime_sec", 0.0) + duration_sec
        pdf_runtime = self._ledger.get("pdf_runtime_sec", {}).get(self._record.pdf, 0.0) + duration_sec
        self._record.status = status
        self._record.classification = classification
        self._record.duration_sec = duration_sec
        self._record.cumulative_runtime_sec = cumulative_runtime
        self._record.pdf_runtime_sec = pdf_runtime
        self._record.outputs = outputs or []
        self._record.logs = logs or []
        self._record.notes = notes
        self._write_per_run_record()
        attempts = self._ledger.setdefault("attempts", [])
        attempts.append(asdict(self._record))
        self._ledger["cumulative_runtime_sec"] = cumulative_runtime
        self._ledger.setdefault("pdf_attempts", {})[self._record.pdf] = self._pdf_attempt_id
        self._ledger.setdefault("pdf_runtime_sec", {})[self._record.pdf] = pdf_runtime
        self._write_ledger()

    def _append_record(self, fields: Dict[str, Any]) -> None:
        temp_record = asdict(self._record)
        temp_record.update(fields)
        self._record = AttemptRecord(**temp_record)
        self._write_per_run_record()

    def _write_per_run_record(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        attempt_log = self.run_dir / "attempt_log.json"
        attempt_log.write_text(json.dumps(asdict(self._record), indent=2), encoding="utf-8")

    def _load_ledger(self) -> Dict[str, Any]:
        if not LEDGER_PATH.exists():
            return {"attempts": [], "cumulative_runtime_sec": 0.0, "pdf_attempts": {}, "pdf_runtime_sec": {}}
        try:
            return json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
        except Exception:
            # Corrupted ledger; archive old file and start fresh.
            backup = LEDGER_PATH.with_suffix(".corrupt")
            os.replace(LEDGER_PATH, backup)
            return {"attempts": [], "cumulative_runtime_sec": 0.0, "pdf_attempts": {}, "pdf_runtime_sec": {}}

    def _write_ledger(self) -> None:
        LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
        LEDGER_PATH.write_text(json.dumps(self._ledger, indent=2), encoding="utf-8")

    @staticmethod
    def can_run_next() -> bool:
        if not LEDGER_PATH.exists():
            return True
        try:
            ledger = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
        except Exception:
            return True
        runtime = float(ledger.get("cumulative_runtime_sec", 0.0) or 0.0)
        return runtime < RUNTIME_CAP_SECONDS
