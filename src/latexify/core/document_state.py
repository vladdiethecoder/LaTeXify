"""
Document state management for the LaTeXify pipeline.

This module provides immutable data structures for tracking pipeline state,
enabling better debugging, caching, and orchestration.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional
from enum import Enum


class ProcessingStage(Enum):
    """Pipeline processing stages."""
    INGESTION = "ingestion"
    LAYOUT_DETECTION = "layout_detection"
    EXTRACTION = "extraction"
    READING_ORDER = "reading_order"
    ASSEMBLY = "assembly"
    REFINEMENT = "refinement"
    COMPILATION = "compilation"
    COMPLETE = "complete"


@dataclass(frozen=True)
class BoundingBox:
    """Immutable bounding box representation."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 1.0
    
    @property
    def width(self) -> float:
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        return self.y2 - self.y1
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


@dataclass(frozen=True)
class LayoutRegion:
    """Detected layout region with classification."""
    bbox: BoundingBox
    category: str  # "Text", "Equation_Display", "Table", "Figure", etc.
    page_num: int
    region_id: str
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExtractionResult:
    """Result from an extraction agent."""
    region_id: str
    content: str
    confidence: float
    extractor_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentState:
    """
    Central state object tracking the entire pipeline execution.
    
    This is intentionally NOT frozen to allow incremental updates as the
    pipeline progresses. However, internal data structures (like layout_manifest)
    should be treated as immutable once populated.
    
    Attributes:
        pdf_path: Path to the source PDF
        image_paths: List of paths to rasterized page images
        layout_manifest: Dictionary mapping page numbers to detected regions
        extractions: Dictionary mapping region_id to extraction results
        metadata: Document-level metadata (title, authors, etc.)
        current_stage: Current pipeline processing stage
        errors: List of errors encountered during processing
        cache_dir: Directory for caching intermediate results
    """
    
    pdf_path: Path
    image_paths: List[Path] = field(default_factory=list)
    layout_manifest: Dict[int, List[LayoutRegion]] = field(default_factory=dict)
    extractions: Dict[str, ExtractionResult] = field(default_factory=dict)
    metadata: Dict[str, str] = field(default_factory=dict)
    current_stage: ProcessingStage = ProcessingStage.INGESTION
    errors: List[Dict[str, Any]] = field(default_factory=list)
    cache_dir: Optional[Path] = None
    
    def add_layout_regions(self, page_num: int, regions: List[LayoutRegion]):
        """Add detected layout regions for a specific page."""
        if page_num not in self.layout_manifest:
            self.layout_manifest[page_num] = []
        self.layout_manifest[page_num].extend(regions)
    
    def add_extraction(self, region_id: str, result: ExtractionResult):
        """Add an extraction result."""
        self.extractions[region_id] = result
    
    def get_regions_by_category(self, category: str) -> List[LayoutRegion]:
        """Get all regions of a specific category across all pages."""
        regions = []
        for page_regions in self.layout_manifest.values():
            regions.extend([r for r in page_regions if r.category == category])
        return regions
    
    def get_regions_for_page(self, page_num: int) -> List[LayoutRegion]:
        """Get all regions for a specific page."""
        return self.layout_manifest.get(page_num, [])
    
    def advance_stage(self, stage: ProcessingStage):
        """Advance to the next pipeline stage."""
        self.current_stage = stage
    
    def log_error(self, stage: str, error: str, details: Optional[Dict[str, Any]] = None):
        """Log an error encountered during processing."""
        error_entry = {
            "stage": stage,
            "error": error,
            "details": details or {}
        }
        self.errors.append(error_entry)
    
    @property
    def total_regions(self) -> int:
        """Total number of detected layout regions."""
        return sum(len(regions) for regions in self.layout_manifest.values())
    
    @property
    def total_pages(self) -> int:
        """Total number of pages."""
        return len(self.image_paths)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dictionary for logging/debugging."""
        return {
            "pdf_path": str(self.pdf_path),
            "total_pages": self.total_pages,
            "total_regions": self.total_regions,
            "extractions_count": len(self.extractions),
            "current_stage": self.current_stage.value,
            "errors_count": len(self.errors),
            "metadata": self.metadata
        }
