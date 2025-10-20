from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class Block:
    bbox: Optional[list] = None      # [x1,y1,x2,y2] if available
    type: Optional[str] = None       # "text" | "table" | "equation" | "figure" | ...
    text: Optional[str] = None
    conf: Optional[float] = None

@dataclass
class OCRPage:
    model: str
    page: int
    text_md: str
    blocks: List[Block]

class OCRBackend:
    name: str = "base"
    def recognize_page(self, image_path: str, page_num: int = 1) -> OCRPage:
        raise NotImplementedError
