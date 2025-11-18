"""Compilation validation and error feedback."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import os
from pathlib import Path
from typing import Dict, List

from ..models.vllm_client import get_vllm_client

ERROR_RE = re.compile(r"! LaTeX Error: (?P<message>.+)")
UNDEF_RE = re.compile(r"Undefined control sequence")
CITATION_WARNING_RE = re.compile(r"LaTeX Warning: Citation")
UNDEFINED_REFERENCE_RE = re.compile(r"There were undefined references")

DEFAULT_VALIDATION_MODEL = os.environ.get("LATEXIFY_VALIDATION_MODEL", "Qwen/Qwen2.5-7B-Instruct")


def run_validation(tex_path: Path, output_path: Path | None = None) -> Dict[str, object]:
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
    if errors and log_text:
        analysis = _summarize_compile_errors(errors, log_text)
    result = {"success": success, "errors": errors, "log_path": str(log_path)}
    if analysis:
        result["analysis"] = analysis
    target = output_path or tex_path.parent.joinpath("validation.json")
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


__all__ = ["run_validation"]


def _summarize_compile_errors(errors: List[str], log_text: str) -> str | None:
    client = get_vllm_client(model=DEFAULT_VALIDATION_MODEL)
    if client is None:
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
    try:  # pragma: no cover - depends on local model
        summary = client.generate(prompt, max_tokens=240)
    except Exception:
        return None
    return summary.strip() or None
