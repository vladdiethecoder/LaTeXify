from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class LayoutBlock:
    id: str
    bbox: List[float] # [x1, y1, x2, y2]
    category: str # text, table, equation, etc.
    confidence: float
    page_num: int
    content: Optional[str] = None # The recognized text/latex
    
    @property
    def center(self) -> Tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2)

class ReadingOrder:
    def sort(self, blocks: List[LayoutBlock]) -> List[LayoutBlock]:
        """
        Sort blocks into reading order.
        Default implementation: Top-down, Left-to-right (XY sort).
        """
        # Simple heuristic: sort by Y primarily, then X.
        # To handle columns, one might need more complex logic (XY Cut).
        # For v1, we'll use a robust row-major sort with a threshold.
        
        return sorted(blocks, key=lambda b: (b.page_num, b.bbox[1], b.bbox[0]))

class GNNReadingOrder(ReadingOrder):
    def __init__(self, model_path: str = None):
        # TODO: Load GNN model
        pass

    def sort(self, blocks: List[LayoutBlock]) -> List[LayoutBlock]:
        """
        Use GNN to predict edge connections and topological sort.
        """
        # 1. Build Graph from blocks (nodes)
        # 2. Predict edges
        # 3. Topological Sort
        # For now, fallback to heuristic
        return super().sort(blocks)
