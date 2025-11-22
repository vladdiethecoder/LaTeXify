from __future__ import annotations

import logging
from typing import Callable, Dict
from .state import DocumentState, ProcessingStatus

# Import nodes
from latexify.pipeline.ingestion_mineru import ingest_node
from latexify.pipeline.planner import plan_node
from latexify.agents.retrieval import retrieve_node
from latexify.pipeline.synthesis import synthesis_node
from latexify.agents.verifier import verification_node
from latexify.agents.refinement import refine_node

LOGGER = logging.getLogger(__name__)

class WorkflowGraph:
    def __init__(self):
        self.nodes: Dict[str, Callable[[DocumentState], DocumentState]] = {}
        
    def add_node(self, name: str, func: Callable[[DocumentState], DocumentState]):
        self.nodes[name] = func
        
    def run(self, initial_state: DocumentState) -> DocumentState:
        state = initial_state
        state.status = ProcessingStatus.PROCESSING
        
        try:
            # Phase 1: Ingestion
            LOGGER.info(">>> Starting Node: ingest_document")
            if "ingest_document" in self.nodes:
                state = self.nodes["ingest_document"](state)
            
            # Phase 2: Planning
            LOGGER.info(">>> Starting Node: plan_structure")
            if "plan_structure" in self.nodes:
                state = self.nodes["plan_structure"](state)
            
            # Phase 3: Retrieval
            LOGGER.info(">>> Starting Node: retrieve_context")
            if "retrieve_context" in self.nodes:
                state = self.nodes["retrieve_context"](state)
                
            # Phase 3.5: Synthesis (Generation)
            LOGGER.info(">>> Starting Node: generate_latex")
            if "generate_latex" in self.nodes:
                state = self.nodes["generate_latex"](state)
            
            # Phase 4: Loop (Compile -> Verify -> Refine)
            max_retries = state.config.get("refinement_passes", 1)
            if state.config.get("skip_compile"):
                LOGGER.info("Skipping compilation as requested.")
            else:
                for attempt in range(max_retries + 1):
                    LOGGER.info(f">>> Starting Node: compile_and_verify (Attempt {attempt+1})")
                    if "compile_and_verify" in self.nodes:
                        state = self.nodes["compile_and_verify"](state)
                    
                    if not state.diagnostics:
                        LOGGER.info("Verification passed.")
                        break
                    
                    if attempt < max_retries:
                        LOGGER.info(f">>> Starting Node: refine_errors (Diagnostics: {len(state.diagnostics)})")
                        if "refine_errors" in self.nodes:
                            state = self.nodes["refine_errors"](state)
                    else:
                        LOGGER.warning("Max refinement retries reached.")
            
            state.status = ProcessingStatus.COMPLETED
            
        except Exception as e:
            LOGGER.error(f"Workflow failed: {e}", exc_info=True)
            state.status = ProcessingStatus.FAILED
            state.add_log(f"Critical Failure: {e}")
            
        return state

def create_workflow() -> WorkflowGraph:
    wf = WorkflowGraph()
    wf.add_node("ingest_document", ingest_node)
    wf.add_node("plan_structure", plan_node)
    wf.add_node("retrieve_context", retrieve_node)
    wf.add_node("generate_latex", synthesis_node)
    wf.add_node("compile_and_verify", verification_node)
    wf.add_node("refine_errors", refine_node)
    return wf