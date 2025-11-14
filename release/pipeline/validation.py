"""Compilation validation and error feedback."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List

ERROR_RE = re.compile(r"! LaTeX Error: (?P<message>.+)")
UNDEF_RE = re.compile(r"Undefined control sequence")


def run_validation(tex_path: Path) -> Dict[str, object]:
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
    if log_path.exists():
        log_text = log_path.read_text(errors="ignore")
        errors.extend(ERROR_RE.findall(log_text))
        if UNDEF_RE.search(log_text):
            errors.append("undefined-control-sequence")
    result = {"success": success, "errors": errors, "log_path": str(log_path)}
    tex_path.parent.joinpath("validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


__all__ = ["run_validation"]
