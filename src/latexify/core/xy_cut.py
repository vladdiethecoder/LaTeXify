"""
XY-Cut Reading Order Algorithm for multi-column layouts.

This module implements recursive XY-Cut for determining the reading order
of layout regions, with special handling for multi-column documents.
"""

import logging
from typing import List, Tuple
from dataclasses import dataclass

from latexify.core.document_state import LayoutRegion, BoundingBox
from latexify.exceptions import ReadingOrderError

logger = logging.getLogger(__name__)


@dataclass
class LayoutBlock:
    """Legacy compatibility class."""
    id: str
    bbox: List[float]
    category: str
    confidence: float
    page_num: int
    content: str


class XYCutReadingOrder:
    """
    XY-Cut algorithm for reading order detection.
    
    Recursively partitions the page into columns and rows to determine
    the natural reading order of content blocks.
    """
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.min_partition_size = self.config.get("min_partition_size", 100)
        self.whitespace_threshold = self.config.get("whitespace_threshold", 50)
        
    def sort(self, regions: List[LayoutRegion]) -> List[LayoutRegion]:
        """
        Sort regions using XY-Cut algorithm.
        
        Args:
            regions: List of LayoutRegion objects
            
        Returns:
            Sorted list of LayoutRegion objects in reading order
        """
        if not regions:
            return []
        
        try:
            return self._xy_cut(regions)
        except Exception as e:
            logger.error(f"XY-Cut failed: {e}. Falling back to naive top-down sort.")
            return self._naive_sort(regions)
    
    def _xy_cut(self, regions: List[LayoutRegion]) -> List[LayoutRegion]:
        """
        Recursive XY-Cut implementation.
        
        Args:
            regions: Regions to partition
            
        Returns:
            Sorted regions
        """
        if len(regions) <= 1:
            return regions
        
        # Get bounding box of all regions
        min_x = min(r.bbox.x1 for r in regions)
        max_x = max(r.bbox.x2 for r in regions)
        min_y = min(r.bbox.y1 for r in regions)
        max_y = max(r.bbox.y2 for r in regions)
        
        width = max_x - min_x
        height = max_y - min_y
        
        # Determine split direction (vertical first for columns)
        if width > height:
            # Try vertical split (columns)
            split_x = self._find_vertical_split(regions, min_x, max_x)
            if split_x is not None:
                left = [r for r in regions if r.bbox.center[0] < split_x]
                right = [r for r in regions if r.bbox.center[0] >= split_x]
                
                # Recursively sort each partition
                return self._xy_cut(left) + self._xy_cut(right)
        
        # Try horizontal split (rows)
        split_y = self._find_horizontal_split(regions, min_y, max_y)
        if split_y is not None:
            top = [r for r in regions if r.bbox.center[1] < split_y]
            bottom = [r for r in regions if r.bbox.center[1] >= split_y]
            
            # Recursively sort each partition
            return self._xy_cut(top) + self._xy_cut(bottom)
        
        # No good split found, use naive sort
        return self._naive_sort(regions)
    
    def _find_vertical_split(
        self,
        regions: List[LayoutRegion],
        min_x: float,
        max_x: float
    ) -> float | None:
        """
        Find best vertical split line (for columns).
        
        Looks for the widest whitespace gap between regions.
        
        Returns:
            X-coordinate of split, or None if no good split
        """
        if len(regions) < 2:
            return None
        
        # Get all vertical boundaries (x1, x2)
        boundaries = []
        for r in regions:
            boundaries.append((r.bbox.x2, 'end'))    # Right edge
            boundaries.append((r.bbox.x1, 'start'))  # Left edge
        
        boundaries.sort()
        
        # Find widest gap
        max_gap = 0
        max_gap_center = None
        
        for i in range(len(boundaries) - 1):
            x1, type1 = boundaries[i]
            x2, type2 = boundaries[i + 1]
            
            gap = x2 - x1
            
            # Only consider gaps between "end" and "start"
            if type1 == 'end' and type2 == 'start' and gap > max_gap:
                max_gap = gap
                max_gap_center = (x1 + x2) / 2
        
        # Only split if gap is significant
        if max_gap > self.whitespace_threshold:
            return max_gap_center
        
        return None
    
    def _find_horizontal_split(
        self,
        regions: List[LayoutRegion],
        min_y: float,
        max_y: float
    ) -> float | None:
        """
        Find best horizontal split line (for rows).
        
        Returns:
            Y-coordinate of split, or None if no good split
        """
        if len(regions) < 2:
            return None
        
        # Get all horizontal boundaries (y1, y2)
        boundaries = []
        for r in regions:
            boundaries.append((r.bbox.y2, 'end'))    # Bottom edge
            boundaries.append((r.bbox.y1, 'start'))  # Top edge
        
        boundaries.sort()
        
        # Find widest gap
        max_gap = 0
        max_gap_center = None
        
        for i in range(len(boundaries) - 1):
            y1, type1 = boundaries[i]
            y2, type2 = boundaries[i + 1]
            
            gap = y2 - y1
            
            # Only consider gaps between "end" and "start"
            if type1 == 'end' and type2 == 'start' and gap > max_gap:
                max_gap = gap
                max_gap_center = (y1 + y2) / 2
        
        # Only split if gap is significant
        if max_gap > self.whitespace_threshold:
            return max_gap_center
        
        return None
    
    def _naive_sort(self, regions: List[LayoutRegion]) -> List[LayoutRegion]:
        """Fallback: simple top-to-bottom, left-to-right sort."""
        return sorted(regions, key=lambda r: (r.bbox.y1, r.bbox.x1))


# Backward compatibility
class ReadingOrder:
    """Legacy wrapper for existing pipeline code."""
    
    def __init__(self, config: dict = None):
        self.xy_cut = XYCutReadingOrder(config)
    
    def sort(self, blocks: List[LayoutBlock]) -> List[LayoutBlock]:
        """
        Sort LayoutBlock objects in reading order.
        
        Args:
            blocks: List of legacy LayoutBlock objects
            
        Returns:
            Sorted list
        """
        # Convert LayoutBlock to LayoutRegion
        regions = []
        for block in blocks:
            bbox = BoundingBox(
                x1=block.bbox[0],
                y1=block.bbox[1],
                x2=block.bbox[2],
                y2=block.bbox[3],
                confidence=block.confidence
            )
            region = LayoutRegion(
                bbox=bbox,
                category=block.category,
                page_num=block.page_num,
                region_id=block.id,
                confidence=block.confidence
            )
            regions.append(region)
        
        # Sort with XY-Cut
        sorted_regions = self.xy_cut.sort(regions)
        
        # Convert back to LayoutBlock
        sorted_blocks = []
        for region in sorted_regions:
            block = LayoutBlock(
                id=region.region_id,
                bbox=[
                    region.bbox.x1,
                    region.bbox.y1,
                    region.bbox.x2,
                    region.bbox.y2
                ],
                category=region.category,
                confidence=region.confidence,
                page_num=region.page_num,
                content=""  # Will be filled by pipeline
            )
            sorted_blocks.append(block)
        
        return sorted_blocks
