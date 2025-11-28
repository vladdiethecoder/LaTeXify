from pathlib import Path
from latexify.core.pipeline import PipelineStep, DocumentState
from latexify.core.ocr_engine import OCREngine
from latexify.core.image_store import PageImageStore
from latexify.config import settings

class OCRStep(PipelineStep):
    def __init__(self, mode: str = "auto"):
        super().__init__("OCR")
        self.mode = mode

    async def process(self, state: DocumentState) -> DocumentState:
        cache_dir = state.file_path.parent / ".latexify_cache"
        
        page_store = PageImageStore(
            pdf_path=state.file_path, 
            cache_dir=cache_dir / "page_rasters", 
            enabled=True
        )
        
        try:
            import fitz
            doc = fitz.open(str(state.file_path))
            total_pages = doc.page_count
            doc.close()
        except Exception as e:
            state.add_log(f"Could not determine page count for OCR: {e}")
            return state

        # Use new OCREngine
        engine = OCREngine(mode=self.mode)
        
        for i in range(total_pages):
            image_path = page_store.get_page_image(i)
            if not image_path:
                state.add_log(f"Skipping OCR for page {i+1}: raster failed")
                continue
                
            result = engine.process_page(i, image_path)
            if result.sources:
                state.ocr_content[i] = dict(result.sources)
                state.add_log(f"OCR page {i+1}: captured {list(result.sources.keys())}")
        
        engine.shutdown()
        return state
