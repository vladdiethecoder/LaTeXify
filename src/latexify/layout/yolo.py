from pathlib import Path
from typing import List, Dict, Any
from ultralytics import YOLO
from .base import LayoutEngine

class YOLOLayoutEngine(LayoutEngine):
    def __init__(self, model_path: str = "yolov10n.pt"):
        """
        Initialize YOLO layout engine.
        
        Args:
            model_path: Path to the YOLO model weights. 
                        Defaults to 'yolov10n.pt' (will download if not found).
        """
        self.model = YOLO(model_path)
        
    def detect(self, image_path: Path) -> List[Dict[str, Any]]:
        results = self.model(source=str(image_path), verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                b = box.xyxy[0].tolist()  # get box coordinates in (top, left, bottom, right) format
                c = box.cls.item()
                conf = box.conf.item()
                label = self.model.names[int(c)]
                
                detections.append({
                    "bbox": b,
                    "class": label,
                    "confidence": conf
                })
                
        return detections
