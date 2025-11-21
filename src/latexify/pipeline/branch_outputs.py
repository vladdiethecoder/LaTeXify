"""Helpers for managing branch-specific output artifacts."""
from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

from .branch_orchestrator import BranchRunResult

LOGGER = logging.getLogger(__name__)

PDF_PLACEHOLDER_BYTES = b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]/Contents 4 0 R>>endobj\n4 0 obj<</Length 0>>stream\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f \n0000000010 00000 n \n0000000061 00000 n \n0000000118 00000 n \n0000000203 00000 n \ntrailer<</Root 1 0 R/Size 5>>\nstartxref\n255\n%%EOF\n"


@dataclass
class BranchOutput:
    name: str
    tex_path: Path
    pdf_path: Path
    sample_name: str
    status: str = "pending"
    metadata: Dict[str, object] = field(default_factory=dict)


class BranchOutputManager:
    """Orchestrates branch-specific output directories and metadata."""

    def __init__(
        self,
        *,
        run_dir: Path,
        reports_dir: Path,
        rel_path_fn: Optional[Callable[[Path], str]] = None,
    ) -> None:
        self.run_dir = run_dir
        self.reports_dir = reports_dir
        self.rel_path_fn = rel_path_fn or (lambda path: str(path))
        self._entries: Dict[str, BranchOutput] = {}

    def register_branch_output(
        self,
        *,
        name: str,
        tex_source: Path | None,
        pdf_source: Path | None,
        status: str,
        metadata: Dict[str, object] | None = None,
        annotate: bool = True,
    ) -> BranchOutput:
        branch_dir = self.run_dir / name
        branch_dir.mkdir(parents=True, exist_ok=True)
        suffix = _branch_suffix(name)
        target_tex = branch_dir / "main.tex"
        target_pdf = branch_dir / f"sample_{suffix}.pdf"
        if tex_source and tex_source.exists():
            _copy_with_optional_annotation(tex_source, target_tex, name if annotate else None)
        else:
            placeholder = _placeholder_tex(name)
            target_tex.write_text(placeholder, encoding="utf-8")
        if pdf_source and pdf_source.exists():
            shutil.copy2(pdf_source, target_pdf)
        else:
            _write_placeholder_pdf(target_pdf)
        entry = BranchOutput(
            name=name,
            tex_path=target_tex,
            pdf_path=target_pdf,
            sample_name=target_pdf.name,
            status=status,
            metadata=metadata or {},
        )
        self._entries[name] = entry
        return entry

    def finalize(
        self,
        *,
        best_branch: str,
        legacy_tex: Path,
        legacy_pdf: Path | None,
    ) -> Path:
        best_entry = self._entries.get(best_branch)
        if best_entry is None:
            LOGGER.warning("Branch %s missing; defaulting to legacy outputs", best_branch)
        else:
            shutil.copy2(best_entry.tex_path, legacy_tex)
            if legacy_pdf and best_entry.pdf_path.exists():
                shutil.copy2(best_entry.pdf_path, legacy_pdf)
        manifest_path = self.reports_dir / "branch_outputs.json"
        manifest = {
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "best_branch": best_branch,
            "entries": {
                name: {
                    "status": entry.status,
                    "tex": self.rel_path_fn(entry.tex_path),
                    "pdf": self.rel_path_fn(entry.pdf_path),
                    "metadata": entry.metadata,
                }
                for name, entry in self._entries.items()
            },
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest_path

    def metrics_summary(self) -> Dict[str, Dict[str, object]]:
        summary: Dict[str, Dict[str, object]] = {}
        for name, entry in self._entries.items():
            summary[name] = {
                "status": entry.status,
                "tex": self.rel_path_fn(entry.tex_path),
                "pdf": self.rel_path_fn(entry.pdf_path),
            }
        return summary


def select_best_branch(results: Iterable[BranchRunResult]) -> str:
    priority = ["branch_c", "branch_b", "branch_a"]
    completed = {result.branch: result for result in results if result.status == "completed"}
    for candidate in priority:
        if candidate in completed:
            return candidate
    return priority[-1]


def _branch_suffix(name: str) -> str:
    if name.startswith("branch_") and len(name) > 7:
        return name.split("_", 1)[1]
    return name.replace("branch", "") or "x"


def _placeholder_tex(name: str) -> str:
    return (
        "% Placeholder for "
        + name
        + "\\n% No branch-specific LaTeX was generated because the upstream stage was skipped."
    )


def _write_placeholder_pdf(target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(PDF_PLACEHOLDER_BYTES)


def _copy_with_optional_annotation(source: Path, target: Path, branch_name: str | None) -> None:
    shutil.copy2(source, target)
    if not branch_name:
        return
    try:
        content = target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return
    header = f"% Branch output: {branch_name}\n"
    if content.startswith(header):
        return
    target.write_text(header + content, encoding="utf-8")


__all__ = ["BranchOutputManager", "select_best_branch"]
