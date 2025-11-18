"""Structured logging utilities for per-run data pathway tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import json


ISO_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"
MAX_NOTES_LEN = 200


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _coerce_list(values: Optional[Iterable[str]]) -> List[str]:
    if not values:
        return []
    return [str(value) for value in values if value is not None]


@dataclass
class DataPathwayLogger:
    """Append-only logger that writes JSONL events and a Markdown narrative."""

    run_id: str
    run_dir: Path
    config: Dict[str, Any] | None = None
    events_path: Path = field(init=False)
    markdown_path: Path = field(init=False)
    _events_cache: List[Dict[str, Any]] = field(default_factory=list, init=False)
    _run_summary: Dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.run_dir / "DATA_PATHWAY.llm.jsonl"
        self.markdown_path = self.run_dir / "DATA_PATHWAY.md"
        if self.config:
            self.set_run_summary({"config": self.config})

    def log_event(self, event: Dict[str, Any]) -> None:
        payload = event.copy()
        payload.setdefault("timestamp_utc", _now_utc())
        payload.setdefault("run_id", self.run_id)
        payload.setdefault("stage", "unknown")
        payload.setdefault("status", "unknown")
        payload["input_files"] = _coerce_list(payload.get("input_files"))
        payload["output_files"] = _coerce_list(payload.get("output_files"))
        payload["models"] = _coerce_list(payload.get("models"))
        notes = str(payload.get("notes") or "")
        if len(notes) > MAX_NOTES_LEN:
            notes = notes[: MAX_NOTES_LEN - 3] + "..."
        payload["notes"] = notes
        try:
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self._events_cache.append(payload)
        except Exception:
            # Structured logging should never abort the main pipeline.
            pass

    def set_run_summary(self, summary: Dict[str, Any]) -> None:
        if not summary:
            return
        self._run_summary.update(summary)

    def write_human_markdown(self) -> None:
        lines = [
            f"# Data Pathway — Run {self.run_id}",
            "",
            f"- Generated: {_now_utc()}",
        ]
        if self._run_summary:
            status = self._run_summary.get("status")
            if status:
                lines.append(f"- Status: **{status.upper()}**")
            input_pdf = self._run_summary.get("input_pdf")
            if input_pdf:
                lines.append(f"- Input PDF: `{input_pdf}`")
            tex_path = self._run_summary.get("tex_path")
            pdf_path = self._run_summary.get("pdf_path")
            if tex_path:
                lines.append(f"- LaTeX Output: `{tex_path}`")
            if pdf_path:
                lines.append(f"- PDF Output: `{pdf_path}`")
            duration = self._run_summary.get("duration_sec")
            if duration is not None:
                lines.append(f"- Duration: {duration:.2f}s")
            run_dir = self._run_summary.get("artifact_dir")
            if run_dir:
                lines.append(f"- Artifact Directory: `{run_dir}`")
            lines.append("")
        if self.config:
            lines.extend(
                [
                    "## Configuration",
                    "",
                    "```json",
                    json.dumps(self.config, indent=2),
                    "```",
                    "",
                ]
            )
        lines.append("## Stage Timeline")
        lines.append("")
        if not self._events_cache:
            lines.append("_No events recorded._")
        else:
            for event in self._events_cache:
                stage = event.get("stage", "unknown")
                status = event.get("status", "unknown")
                notes = event.get("notes") or ""
                outputs = event.get("output_files") or []
                output_suffix = f" → {', '.join(f'`{path}`' for path in outputs)}" if outputs else ""
                lines.append(f"- **{stage}** `{status}` {notes}{output_suffix}".rstrip())
        self.markdown_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def init_logger(run_id: str, run_dir: Path, config: Dict[str, Any] | None = None) -> DataPathwayLogger:
    """Factory helper so callers avoid importing the class directly."""

    return DataPathwayLogger(run_id=run_id, run_dir=run_dir, config=config or {})


__all__ = ["DataPathwayLogger", "init_logger"]
