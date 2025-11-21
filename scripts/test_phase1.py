import sys
from pathlib import Path

# Add src to sys.path if not installed
project_root = Path(__file__).resolve().parents[1]
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

from latexify.ingestion.pymupdf import PyMuPDFIngestor
from latexify.layout.yolo import YOLOLayoutEngine

def main():
    input_pdf = project_root / "src/latexify/inputs/sample.pdf"
    output_dir = project_root / "data/test_output"
    
    if not input_pdf.exists():
        print(f"Sample PDF not found at {input_pdf}")
        # Create a dummy PDF if missing? Or assume it exists as per ls output
        return

    print(f"Ingesting {input_pdf}...")
    ingestor = PyMuPDFIngestor(dpi=300)
    try:
        images = ingestor.ingest(input_pdf, output_dir=output_dir)
        print(f"Generated {len(images)} images in {output_dir}")
    except Exception as e:
        print(f"Ingestion failed: {e}")
        return

    print("Initializing Layout Engine (YOLO)...")
    # Using yolov8n.pt for now as it is standard in ultralytics
    # User should update to yolov10n.pt when available or provide path
    model_name = "yolov8n.pt" 
    try:
        layout_engine = YOLOLayoutEngine(model_path=model_name) 
        
        for img_path in images:
            print(f"Processing {img_path.name}...")
            detections = layout_engine.detect(img_path)
            print(f"Detected {len(detections)} objects:")
            for det in detections:
                print(f" - {det['class']} ({det['confidence']:.2f}) at {det['bbox']}")
    except Exception as e:
        print(f"Layout detection failed: {e}")

if __name__ == "__main__":
    main()
