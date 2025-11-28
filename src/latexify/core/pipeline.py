"""
Core LaTeXify pipeline orchestrator.

This module defines the high-level `LaTeXifyPipeline` façade used by CLIs and
APIs. It adapts a typed ``RuntimeConfig`` into the `DocumentState` consumed by
the underlying workflow graph.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

from latexify.config import RuntimeConfig
from latexify.core.state import DocumentState, ProcessingStatus
from latexify.core.workflow import create_workflow

logger: Final = logging.getLogger(__name__)


class LaTeXifyPipeline:
    """Typed façade around the workflow graph."""

    def __init__(self, cfg: RuntimeConfig) -> None:
        self._cfg = cfg
        self._workflow = create_workflow()

    def process(self, pdf_path: Path, *, skip_compile: bool = False) -> str:
        """Run the end-to-end PDF→LaTeX pipeline."""
        logger.info("Initializing Pipeline for %s", pdf_path)

        ingestion = self._cfg.pipeline.ingestion
        refinement = self._cfg.pipeline.refinement
        hardware = self._cfg.hardware

        flat_config = {
            "chunk_chars": ingestion.chunk_chars,
            "ingestion_backend": ingestion.backend,
            "ingestion_dpi": ingestion.dpi,
            "docling_options": ingestion.docling,
            "use_vllm": refinement.use_vllm,
            "llm_repo": refinement.llm_repo,
            "llm_device": hardware.llm_device,
            "load_in_4bit": refinement.load_in_4bit,
            "load_in_8bit": refinement.load_in_8bit,
            "refinement_passes": refinement.refinement_passes,
            "skip_compile": skip_compile,
            "enable_formal_verification": False,
        }

        state = DocumentState(
            document_name=pdf_path.stem.replace("_", " ").title(),
            file_path=pdf_path,
            config=flat_config,
        )

        logger.info("Executing Workflow...")
        final_state = self._workflow.run(state)

        if final_state.status == ProcessingStatus.FAILED:
            logger.error("Pipeline reported failure.")
            raise RuntimeError(f"Pipeline failed. Logs: {final_state.processing_log}")

        return final_state.generated_latex
