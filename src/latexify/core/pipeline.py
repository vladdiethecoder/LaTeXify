"""
Core LaTeXify Pipeline Orchestrator.
Uses the WorkflowGraph to execute the Phase 1-4 Blueprint.
"""
import logging
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

from latexify.core.state import DocumentState, ProcessingStatus
from latexify.core.workflow import create_workflow

logger = logging.getLogger(__name__)

class LaTeXifyPipeline:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.workflow = create_workflow()

    def process(self, pdf_path: Path) -> str:
        logger.info(f"Initializing Pipeline for {pdf_path}")
        
        pipeline_cfg = OmegaConf.to_container(self.cfg.pipeline, resolve=True)
        hardware_cfg = OmegaConf.to_container(self.cfg.hardware, resolve=True) if "hardware" in self.cfg else {}
        
        flat_config = {
            "chunk_chars": pipeline_cfg.get("ingestion", {}).get("chunk_chars", 2000),
            "use_vllm": pipeline_cfg.get("refinement", {}).get("use_vllm", True),
            "llm_repo": pipeline_cfg.get("refinement", {}).get("llm_repo", None),
            "llm_device": hardware_cfg.get("llm_device", "cuda"),
            "load_in_4bit": pipeline_cfg.get("refinement", {}).get("load_in_4bit", False),
            "load_in_8bit": pipeline_cfg.get("refinement", {}).get("load_in_8bit", False),
            "refinement_passes": 1,
            "skip_compile": False,
            "enable_formal_verification": False
        }
        
        state = DocumentState(
            document_name=pdf_path.stem.replace("_", " ").title(),
            file_path=pdf_path,
            config=flat_config
        )
        
        logger.info("Executing Workflow...")
        final_state = self.workflow.run(state)
        
        if final_state.status == ProcessingStatus.FAILED:
            logger.error("Pipeline reported failure.")
            raise RuntimeError(f"Pipeline failed. Logs: {final_state.processing_log}")
            
        return final_state.generated_latex
