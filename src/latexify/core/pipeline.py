from abc import ABC, abstractmethod
import logging
import asyncio
from typing import List
from .state import DocumentState, ProcessingStatus

class PipelineStep(ABC):
    """
    Abstract handler in the Chain of Responsibility.
    """
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(name)

    @abstractmethod
    async def process(self, state: DocumentState) -> DocumentState:
        """
        Pure logic transformation. 
        Returns the modified state.
        """
        pass

class PipelineRunner:
    def __init__(self, steps: List[PipelineStep]):
        self.steps = steps
        self.logger = logging.getLogger("PipelineRunner")

    async def run(self, initial_state: DocumentState) -> DocumentState:
        current_state = initial_state
        step = None # Initialize step to handle empty steps list edge case though unlikely
        
        try:
            for step in self.steps:
                current_state.status = ProcessingStatus.PROCESSING
                current_state.add_log(f"Starting step: {step.name}")
                self.logger.info(f"Starting step: {step.name}")
                
                # Execute step
                current_state = await step.process(current_state)
                
            current_state.status = ProcessingStatus.COMPLETED
            current_state.add_log("Pipeline completed successfully")
            return current_state
            
        except Exception as e:
            current_state.status = ProcessingStatus.FAILED
            step_name = step.name if step else "Init"
            error_msg = f"Pipeline failed at {step_name}: {str(e)}"
            current_state.add_log(error_msg)
            self.logger.error(error_msg, exc_info=True)
            raise e
