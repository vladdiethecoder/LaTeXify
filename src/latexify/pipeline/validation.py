"""Compilation validation and error feedback."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List

from ..models.kimi_k2_adapter import LATEX_VALIDATION_GRAMMAR, get_kimi_adapter
from .latex_repair_agent import KimiK2LatexRepair

ERROR_RE = re.compile(r"! LaTeX Error: (?P<message>.+)")
UNDEF_RE = re.compile(r"Undefined control sequence")
CITATION_WARNING_RE = re.compile(r"LaTeX Warning: Citation")
UNDEFINED_REFERENCE_RE = re.compile(r"There were undefined references")

VALIDATION_TEMPERATURE = float(os.environ.get("LATEXIFY_VALIDATION_TEMPERATURE", "0.1"))
VALIDATION_MAX_TOKENS = int(os.environ.get("LATEXIFY_VALIDATION_MAX_TOKENS", "240"))


def run_validation(tex_path: Path, output_path: Path | None = None) -> Dict[str, object]:
    repair_agent = KimiK2LatexRepair()
    log_path = tex_path.with_suffix(".log")
    pdf_path = tex_path.with_suffix(".pdf")
    compile_engine = shutil.which("tectonic") or shutil.which("latexmk")
    errors: List[str] = []
    success = False
    if compile_engine:
        cmd = (
            [compile_engine, "--keep-logs", "--keep-intermediates", str(tex_path)]
            if "tectonic" in compile_engine
            else [compile_engine, "-pdf", "-silent", str(tex_path)]
        )
        try:
            subprocess.run(cmd, check=True, cwd=tex_path.parent)
            success = pdf_path.exists()
        except subprocess.CalledProcessError:
            success = False
    analysis: str | None = None
    log_text = ""
    if log_path.exists():
        log_text = log_path.read_text(errors="ignore")
        errors.extend(ERROR_RE.findall(log_text))
        if UNDEF_RE.search(log_text):
            errors.append("undefined-control-sequence")
        if CITATION_WARNING_RE.search(log_text):
            errors.append("undefined-citation")
        if UNDEFINED_REFERENCE_RE.search(log_text):
            errors.append("undefined-reference")
    citations_path = tex_path.parent / "citations.json"
    if citations_path.exists():
        try:
            citation_data = json.loads(citations_path.read_text(encoding="utf-8"))
            if citation_data.get("citations") and not citation_data.get("bibliography"):
                errors.append("bibliography-missing")
        except json.JSONDecodeError:
            errors.append("citations-corrupt")
    result: Dict[str, object] = {"success": success, "errors": errors, "log_path": str(log_path)}
    if errors and log_text:
        analysis = _summarize_compile_errors(errors, log_text)
        diagnostics = repair_agent.diagnose_log(log_text)
        if diagnostics:
            result["diagnostics"] = [
                {"code": entry.code, "detail": entry.detail} for entry in diagnostics
            ]
        semantic = repair_agent.semantic_validate(tex_path)
        if semantic.get("issues"):
            result["semantic_issues"] = semantic["issues"]
        if os.environ.get("LATEXIFY_AUTOREPAIR", "1") != "0":
            repair_summary = repair_agent.repair_from_log(tex_path, log_path)
            if repair_summary.get("actions"):
                result["repair_actions"] = repair_summary["actions"]
    result["success"] = success
    result["errors"] = errors
    if analysis:
        result["analysis"] = analysis
    target = output_path or tex_path.parent.joinpath("validation.json")
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


__all__ = ["run_validation"]


def _summarize_compile_errors(errors: List[str], log_text: str) -> str | None:
    adapter = get_kimi_adapter()
    if adapter is None:
        return None
    truncated = "\n".join(log_text.splitlines()[-80:])
    prompt = (
        "You are a LaTeX diagnostics assistant. Review the compilation errors and explain the root causes "
        "succinctly:\n\n"
        f"Errors: {errors[:5]}\n"
        "Log tail:\n"
        f"{truncated}\n\n"
        "Respond with one paragraph describing the likely fix."
    )
    try:  # pragma: no cover - depends on llama.cpp runtime
        summary = adapter.generate(
            prompt,
            max_tokens=VALIDATION_MAX_TOKENS,
            temperature=VALIDATION_TEMPERATURE,
            grammar=LATEX_VALIDATION_GRAMMAR,
        )
    except Exception:
        return None
    return summary.strip() or None
