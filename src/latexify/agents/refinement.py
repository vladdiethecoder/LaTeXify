"""Refinement Agent Node."""
from __future__ import annotations

import logging
from latexify.core.state import DocumentState
from latexify.refinement.refiner import LLMRefiner

LOGGER = logging.getLogger(__name__)

def refine_node(state: DocumentState) -> DocumentState:
    """
    Refinement Node: Uses LLM to fix compilation errors.
    """
    if not state.diagnostics:
        return state
        
    LOGGER.info("Starting Refinement Node...")
    LOGGER.info(f"Config: llm_repo={state.config.get('llm_repo')}, device={state.config.get('llm_device')}, vllm={state.config.get('use_vllm')}")
    
    # Initialize Refiner
    refiner = LLMRefiner(
        model_path=state.config.get("llm_repo", "Qwen/Qwen2.5-Coder-14B-Instruct"),
        device=state.config.get("llm_device", "cuda"),
        use_vllm=state.config.get("use_vllm", True),
        load_in_4bit=state.config.get("load_in_4bit", False),
        load_in_8bit=state.config.get("load_in_8bit", False)
    )
    
    error_log = "\n".join(state.diagnostics)
    LOGGER.info(f"Attempting to fix {len(state.diagnostics)} errors...")
    
    fixed_latex = refiner.fix_error(state.generated_latex, error_log)
    
    if fixed_latex and len(fixed_latex) > 10: 
        state.generated_latex = fixed_latex
        state.diagnostics = [] 
        state.add_log("Applied LLM refinement fix.")
        LOGGER.info("Refinement applied.")
    else:
        state.add_log("Refinement failed to produce valid output.")
        LOGGER.warning("Refinement failed.")
        
    return state
