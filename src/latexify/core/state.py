from enum import Enum
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict, field_validator

class ProcessingStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class TextBlock(BaseModel):
    text: str
    confidence: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    page_index: int = 0
    tag: str = "text"
    metadata: dict = Field(default_factory=dict)

class DocumentState(BaseModel):
    """
    The Single Source of Truth passed through the pipeline.
    """
    model_config = ConfigDict(strict=True, arbitrary_types_allowed=True)

    # Metadata
    doc_id: str
    file_path: Path
    status: ProcessingStatus = ProcessingStatus.PENDING
    
    # Data Layers (Populated by respective steps)
    raw_text: Optional[str] = None
    pages: List[str] = Field(default_factory=list)
    ocr_blocks: List[TextBlock] = Field(default_factory=list)
    ocr_content: dict[int, dict[str, str]] = Field(default_factory=dict)
    chunks: List[dict] = Field(default_factory=list)
    
    # Observability
    processing_log: List[str] = Field(default_factory=list)

    def add_log(self, message: str):
        self.processing_log.append(message)

    @field_validator('ocr_blocks')
    @classmethod
    def validate_blocks(cls, v):
        # Ensure confidence is logical
        if any(b.confidence < 0 or b.confidence > 1 for b in v):
            raise ValueError("Confidence must be between 0 and 1")
        return v
