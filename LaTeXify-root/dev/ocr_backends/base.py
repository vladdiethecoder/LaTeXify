# dev/ocr_backends/base.py
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

@dataclass
class Block:
    bbox: Optional[List[int]] = None         # [x1,y1,x2,y2] in pixels, if known
    type: str = "text"                       # "text" | "table" | "equation" | "figure"
    text: str = ""
    conf: Optional[float] = None

@dataclass
class OCRResult:
    model: str
    page: int
    text_md: str = ""
    blocks: List[Block] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["blocks"] = [asdict(b) for b in (self.blocks or [])]
        return d

class OCRBackend:
    name: str = "base"
    def recognize_page(self, image_path: str, page: int=1) -> OCRResult:
        raise NotImplementedError
