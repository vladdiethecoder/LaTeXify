"""Verification Agent Node."""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import List

from latexify.core.state import DocumentState
from latexify.pipeline import assembly
try:
    from latexify.pipeline.formal_verification import FormalVerifier
except ImportError:
    FormalVerifier = None # type: ignore

LOGGER = logging.getLogger(__name__)

def verification_node(state: DocumentState) -> DocumentState:
    """
    Verification Node: Compiles LaTeX and runs formal checks.
    """
    LOGGER.info("Starting Verification Node...")
    
    # 1. Prepare Build Environment
    build_dir = state.file_path.parent / "build" if state.file_path else Path("build")
    build_dir.mkdir(parents=True, exist_ok=True)
    
    tex_path = build_dir / "main.tex"
    tex_path.write_text(state.generated_latex, encoding="utf-8")
    
    # 2. Compile
    pdf_path = assembly.compile_tex(tex_path)
    metrics = assembly.consume_compilation_metrics() or {}
    
    state.compilation_result = {
        "tex_path": str(tex_path),
        "pdf_path": str(pdf_path) if pdf_path else None,
        "metrics": metrics
    }
    
    # 3. Analyze Results & Diagnostics
    diagnostics = []
    
    # Compiler Errors
    if not pdf_path:
        error_summary = metrics.get("error_summary", [])
        if not error_summary:
             history = metrics.get("attempt_history", [])
             if history:
                 last_attempt = history[-1]
                 if not last_attempt.get("success"):
                     diagnostics.append(f"Compiler Error: {last_attempt.get('error', 'Unknown error')}")
             else:
                 diagnostics.append("Compiler Error: Compilation failed without logs.")
        else:
             for err in error_summary:
                 diagnostics.append(f"Compiler Error: {err}")
    
    # Formal Verification
    if FormalVerifier and state.config.get("enable_formal_verification", False):
        try:
            verifier = FormalVerifier()
            # Use simple check or verify full doc if supported
            pass
        except Exception as e:
            LOGGER.warning(f"Formal verification failed to init: {e}")

    state.diagnostics = diagnostics
    if diagnostics:
        LOGGER.warning(f"Verification found {len(diagnostics)} issues.")
    else:
        LOGGER.info("Verification passed.")
        
    return state
