from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from .common import Chunk

class ProcessingStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class TextBlock(BaseModel):
    """Represents a layout element (text, figure, etc.) from ingestion."""
    text: str
    confidence: float = 1.0
    bbox: Optional[tuple[float, float, float, float]] = None  # x1, y1, x2, y2
    page_index: int = 0
    tag: str = "text" # e.g., "question", "header", "equation"
    metadata: dict = Field(default_factory=dict)

class DocumentState(BaseModel):
    """
    The Single Source of Truth passed through the pipeline.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Input / Metadata
    document_name: str = "document"
    file_path: Optional[Path] = None
    
    # Configuration (Runtime Toggles)
    config: Dict[str, Any] = Field(default_factory=lambda: {
        "chunk_chars": 2000,
        "use_vllm": True,
        "refinement_passes": 1,
        "skip_compile": False
    })
    
    # Phase 1: Ingestion Data
    # Structured blocks from MinerU/PyMuPDF
    layout_blocks: List[TextBlock] = Field(default_factory=list)
    # Semantic chunks created from layout blocks
    chunks: List[Chunk] = Field(default_factory=list)
    
    # Phase 2: Planning
    # The semantic structure (MasterPlan)
    semantic_plan: Optional[Dict[str, Any]] = None 
    
    # Phase 3: Retrieval & Synthesis
    # Context snippets retrieved for blocks (ChunkID -> List[Snippet])
    reference_snippets: Dict[str, List[Any]] = Field(default_factory=dict)
    
    # The working draft of LaTeX code
    generated_latex: str = ""
    
    # Phase 4: Validation & Repair
    # Artifacts from compilation (PDF path, logs)
    compilation_result: Dict[str, Any] = Field(default_factory=dict)
    
    # Errors/Warnings to drive self-healing loop
    diagnostics: List[str] = Field(default_factory=list)
    
    # Status Tracking
    status: ProcessingStatus = ProcessingStatus.PENDING
    processing_log: List[str] = Field(default_factory=list)

    def add_log(self, message: str):
        self.processing_log.append(message)